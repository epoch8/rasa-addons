from dataclasses import dataclass, field
import json
import logging
from operator import itemgetter
from typing import Any, Deque, Iterable, Set, Text, List, Dict, Optional, Tuple
from uuid import uuid4

from pyvis.network import Network

from rasa.core.constants import MEMOIZATION_POLICY_PRIORITY
from rasa.core.policies.policy import Policy
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.policies.policy import PolicyPrediction, SupportedData
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.events import (
    Event,
    UserUttered,
    SlotSet,
    ActiveLoop,
    ActionExecuted,
    INTENT_NAME_KEY,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import (
    StoryGraph,
    RuleStep,
    STORY_START,
    STORY_END,
    ACTION_LISTEN_NAME,
)

logger = logging.getLogger(__name__)


class EventGraphNode:
    def __init__(
        self,
        event: Text,
        uid: Text,
        type_name: Text | None = None,
        block_name: Text | None = None,
        data: Dict[Text, Any] | None = None,
        debug_data: Dict[Text, Any] | None = None,
    ) -> None:
        self.event = event
        self.uid = uid
        self.type_name = type_name
        self.block_name = block_name
        self.data = data or {}
        self.debug_data = debug_data or {}
        self.parents: List['EventGraphNode'] = []
        self.children: List['EventGraphNode'] = []

    def add_child(self, child_node: 'EventGraphNode') -> None:
        if child_node is self:
            raise RuntimeError(f'Trying add self as child for node {self.event}')
        self.children.append(child_node)
        child_node.parents.append(self)

    @classmethod
    def from_event(
        cls, event: Event, uid: Text, block_name: Text | None = None
    ) -> 'EventGraphNode':
        event_key = event_to_key(event)
        return cls(event_key, uid, event.type_name, block_name, event.as_dict())

    def to_dict(self) -> Dict[Text, Any]:
        return {
            'event': self.event,
            'uid': self.uid,
            'type_name': self.type_name,
            'block_name': self.block_name,
            'data': self.data,
            'parents': [p.uid for p in self.parents],
        }

    @classmethod
    def from_dict(cls, node_dict: Dict[Text, Any]) -> 'EventGraphNode':
        return cls(
            event=node_dict['event'],
            uid=node_dict['uid'],
            type_name=node_dict.get('type_name'),
            block_name=node_dict.get('block_name'),
            data=node_dict.get('data'),
        )

    def __str__(self) -> str:
        d = self.to_dict()
        del d['parents']
        return json.dumps(d)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self)})'


@dataclass
class SearchPath:
    nodes: List[EventGraphNode] = field(default_factory=list)

    def finished(self) -> bool:
        if not self.nodes:
            return True
        return not self.nodes[-1].parents


@dataclass
class NextActionCandidate:
    action_name: Text
    probability: float


def event_to_key(event: Event) -> Text:
    type_name = event.type_name
    if isinstance(event, UserUttered):
        # Todo handle enities
        return f'{type_name}_{event.intent_name}'
    if isinstance(event, ActionExecuted):
        return f'{type_name}_{event.action_name}'
    if isinstance(event, ActiveLoop):
        return f'{type_name}_{event.name}'
    if isinstance(event, SlotSet):
        return f'{type_name}_{event.key}'
    raise ValueError(f'Unsupported event "{event.type_name}" passed to event_to_key()')


INTENT_ANY = UserUttered(intent={INTENT_NAME_KEY: '*'})
INTENT_ANY_KEY = event_to_key(INTENT_ANY)


class EventGraph:
    class ValidationError(RuntimeError): ...

    def __init__(
        self, graph_nodes: Dict[Text, List[EventGraphNode]], validate=True
    ) -> None:
        if validate:
            self.validate(graph_nodes)
        self._graph_nodes = graph_nodes

    @staticmethod
    def validate(graph_nodes: Dict[Text, List[EventGraphNode]]) -> None: ...

    @classmethod
    def from_story_graph(
        cls, story_graph: StoryGraph, domain: Domain | None = None
    ) -> 'EventGraph':
        graph_nodes: Dict[Text, List[EventGraphNode]] = {}
        # logger.debug(f'N steps: {len(story_graph.story_steps)}')
        # logger.debug(f'N checkpoints: {len(story_graph.story_end_checkpoints)}')
        # logger.debug(f'Checkpoints: {story_graph.story_end_checkpoints}')
        steps = story_graph.ordered_steps()
        logger.debug(f'ordered_steps:')
        start_chekpoint_nodes: Dict[Text, List[EventGraphNode]] = {}
        end_chekpoint_nodes: Dict[Text, List[EventGraphNode]] = {}
        i = 0
        for step in steps:
            logger.debug(step.block_name)
            if isinstance(step, RuleStep):
                logger.debug('  Skipping rule')
                continue
            logger.debug(f'  s_cp: {step.start_checkpoints}')
            prev_node: Optional[EventGraphNode] = None
            for event in step.events:
                event_key = event_to_key(event)
                logger.debug(f'      {event_key}')
                graph_node = EventGraphNode.from_event(event, str(i), step.block_name)
                if (
                    domain
                    and isinstance(event, ActionExecuted)
                    and event.action_name
                    and event.action_name.startswith('utter_')
                ):
                    # TODO why empty list?
                    graph_node.debug_data['responses'] = domain.responses.get(
                        event.action_name
                    )

                if prev_node is None:
                    for s_cp in step.start_checkpoints:
                        if s_cp.name == STORY_START:
                            continue
                        start_chekpoint_nodes.setdefault(s_cp.name, []).append(
                            graph_node
                        )
                else:
                    prev_node.add_child(graph_node)

                prev_node = graph_node
                graph_nodes.setdefault(event_key, []).append(graph_node)
                i += 1

            logger.debug(f'  e_cp: {step.end_checkpoints}')
            for e_cp in step.end_checkpoints:
                # TODO handle condition
                if e_cp.name == STORY_END or prev_node is None:
                    continue
                end_chekpoint_nodes.setdefault(e_cp.name, []).append(prev_node)

        # Connect steps
        for cp_name, end_nodes in end_chekpoint_nodes.items():
            for e_node in end_nodes:
                if cp_name not in start_chekpoint_nodes:
                    raise RuntimeError(
                        f'No matching checkpoint {cp_name} '
                        f'found for node {e_node.event}!'
                    )
                for s_node in start_chekpoint_nodes[cp_name]:
                    e_node.add_child(s_node)

        return cls(graph_nodes)

    def all_nodes(self) -> Set[EventGraphNode]:
        return set(
            node for node_list in self._graph_nodes.values() for node in node_list
        )

    def to_dict(self) -> Dict[Text, Dict[Text, Any]]:
        result: Dict[Text, Dict[Text, Any]] = {}
        for nodes_list in self._graph_nodes.values():
            for node in nodes_list:
                if node.uid in result:
                    continue
                result[node.uid] = node.to_dict()
        return result

    @classmethod
    def from_dict(cls, nodes_dict: Dict[Text, Dict[Text, Any]]) -> 'EventGraph':
        graph_nodes: Dict[Text, List[EventGraphNode]] = {}
        indexed_nodes = {
            uid: EventGraphNode.from_dict(node) for uid, node in nodes_dict.items()
        }
        for uid, node in nodes_dict.items():
            event_node = indexed_nodes[uid]
            for parent_uid in node['parents']:
                indexed_nodes[parent_uid].add_child(event_node)
            graph_nodes.setdefault(event_node.event, []).append(event_node)

        return cls(graph_nodes)

    @staticmethod
    def event_node_style_params(event_node: EventGraphNode) -> Dict[Text, Any]:
        if event_node.type_name == 'user':
            return {'shape': 'box', 'color': 'cyan'}
        # if event_node.type_name == 'action':
        #     if event_node.event.startswith('utter_'):
        #         return {'shape': 'box', 'color': 'green'}
        #     return {'shape': 'box', 'color': 'red'}
        return {'shape': 'box'}

    def visualize(self, result_path: Text) -> None:
        net = Network(
            height='100vh',
            width='100%',
            directed=True,
            # filter_menu=True,
            layout={
                'hierarchical': {
                    'enabled': True,
                    'sortMethod': 'directed',
                    'shakeTowards': 'leaves',
                }
            },
        )
        event_nodes = self.all_nodes()
        for node in event_nodes:
            node_label = f'{node.uid}: {node.event}'
            node_info = node.to_dict()
            del node_info['parents']
            node_info['debug'] = node.debug_data
            net.add_node(
                node.uid,
                node_label,
                title=json.dumps(node_info, indent=2),
                **self.event_node_style_params(node),
            )
        for node in event_nodes:
            if not node.parents:
                story_name = f'Story {node.block_name or uuid4()}'
                net.add_node(story_name, story_name, shape='ellipse', color='green')
                net.add_edge(story_name, node.uid, physics=False)
            for parent in node.parents:
                net.add_edge(parent.uid, node.uid, physics=False)
        net.write_html(result_path)

    def _find_nodes_for_event(self, event: Event) -> List[EventGraphNode]:
        nodes: List[EventGraphNode] = list(
            self._graph_nodes.get(event_to_key(event), [])
        )
        if isinstance(event, UserUttered):
            nodes.extend(self._graph_nodes.get(INTENT_ANY_KEY, []))
        return nodes

    def _init_seach_paths(self, event: Event) -> List[SearchPath]:
        return [SearchPath([node]) for node in self._find_nodes_for_event(event)]

    @staticmethod
    def _get_parents_for_comparation(node: EventGraphNode) -> Iterable[EventGraphNode]:
        # Get node parents. If parent represents "slot" event or inside loop, skip it and look for it parents.
        p_nodes: List[EventGraphNode] = []
        nodes_to_check: Set[EventGraphNode] = set(node.parents)
        while nodes_to_check:
            new_nodes_to_check: Set[EventGraphNode] = set()
            for node_to_check in nodes_to_check:
                if node_to_check.type_name == SlotSet.type_name:
                    new_nodes_to_check.update(node_to_check.parents)
                else:
                    p_nodes.append(node_to_check)
            nodes_to_check = new_nodes_to_check
        return set(p_nodes)

    def _update_seach_paths(
        self, search_paths: List[SearchPath], event: Event
    ) -> List[SearchPath]:
        event_key = event_to_key(event)

        def _compare_event_with_node(node: EventGraphNode) -> bool:
            if event_key == node.event:
                return True
            if isinstance(event, UserUttered) and node.event == INTENT_ANY_KEY:
                return True
            return False

        new_search_paths: List[SearchPath] = []
        for sp in search_paths:
            last_node = sp.nodes[-1]
            if not last_node.parents:
                # finished path
                new_search_paths.append(sp)
                continue
            for p_node in self._get_parents_for_comparation(last_node):
                if _compare_event_with_node(p_node):
                    new_search_paths.append(SearchPath(sp.nodes + [p_node]))

        return new_search_paths

    def find_possible_paths(self, events: Deque[Event]) -> List[SearchPath]:
        search_paths: List[SearchPath] = []
        finished_paths: List[SearchPath] = []
        # TODO handle last event
        accepteble_events = set(
            [
                UserUttered.type_name,
                ActionExecuted.type_name,
                ActiveLoop.type_name,
            ]
        )
        active_loop = False
        is_rule_turn = False
        for event in reversed(events):
            if isinstance(event, ActiveLoop):
                active_loop = event.name is None
            elif active_loop:
                continue
            if (event.type_name not in accepteble_events) or (
                isinstance(event, ActionExecuted)
                and (event.action_name == ACTION_LISTEN_NAME)
            ):
                logger.debug(f'Ignoring event {event}')
                continue
            was_rule_turn = is_rule_turn
            is_rule_turn = (
                isinstance(event, ActionExecuted) and event.policy == 'RulePolicy'
            )
            if is_rule_turn or was_rule_turn:
                logger.debug(f'Skipping rule turn event {event.as_dict()}')
                continue
            if not search_paths:
                search_paths = self._init_seach_paths(event)
                logger.debug(
                    f'Initial search nodes: {[sp.nodes[0] for sp in search_paths]}'
                )
            else:
                search_paths = self._update_seach_paths(search_paths, event)
            finished_paths.extend(sp for sp in search_paths if sp.finished())
            search_paths = [sp for sp in search_paths if not sp.finished()]
            if not search_paths:
                break

        return finished_paths

    @staticmethod
    def _check_slot_value(node: EventGraphNode, tracker: DialogueStateTracker) -> bool:
        if node.type_name != SlotSet.type_name:
            return False
        node_slot_name = node.data['name']
        node_slot_value = node.data['value']
        slot = tracker.slots.get(node_slot_name)
        if slot is None:
            return False
        return slot.value == node_slot_value or (
            node_slot_value == 'set' and slot.value is not None
        )

    def _find_possible_next_actions(
        self, current_node_candidate: EventGraphNode, tracker: DialogueStateTracker
    ) -> Iterable[EventGraphNode | Text]:
        possible_next_actions: List[EventGraphNode] = []
        if not current_node_candidate.children:
            return [ACTION_LISTEN_NAME]
        nodes_to_check = set(current_node_candidate.children)
        while nodes_to_check:
            new_nodes_to_check: Set[EventGraphNode] = set()
            for node in nodes_to_check:
                if node.type_name == SlotSet.type_name:
                    if self._check_slot_value(node, tracker):
                        new_nodes_to_check.update(node.children)
                elif node.type_name == UserUttered.type_name:
                    possible_next_actions.append(ACTION_LISTEN_NAME)
                else:
                    possible_next_actions.append(node)

            nodes_to_check = new_nodes_to_check
        return possible_next_actions

    @staticmethod
    def _calculate_probabilities(
        search_results: Iterable[Tuple[SearchPath, Iterable[Text]]],
    ) -> Iterable[NextActionCandidate]:
        def calc_path_weight(path: SearchPath) -> float:
            n_any_intents = 0
            for node in path.nodes:
                if node.event == INTENT_ANY_KEY:
                    n_any_intents += 1
            return 0.9**n_any_intents

        action_candidates = sorted(
            (
                (len(sp.nodes), calc_path_weight(sp), actions)
                for sp, actions in search_results
                if actions
            ),
            key=itemgetter(0),
        )

        if not action_candidates:
            return []

        weights: Dict[Text, List[float]] = {}
        len_w: float = 0
        prev_path_len = action_candidates[0][0]
        for path_len, path_w, actions in action_candidates:
            if path_len > prev_path_len:
                len_w += 1
                prev_path_len = path_len
            final_path_w = len_w + path_w
            for action in actions:
                weights.setdefault(action, []).append(final_path_w)

        final_weights = {action: max(w) for action, w in weights.items()}
        weights_sum = sum(final_weights.values())

        return (
            NextActionCandidate(action, weight / weights_sum)
            for action, weight in final_weights.items()
        )

    @staticmethod
    def _get_action_name(action: EventGraphNode | Text) -> Text:
        if isinstance(action, Text):
            return action
        elif action.type_name == ActionExecuted.type_name:
            return action.data['name']
        raise RuntimeError(f'Invalid event predicted: {action}')

    def find_next_action_candidates(
        self, tracker: DialogueStateTracker
    ) -> Iterable[NextActionCandidate]:
        if tracker.active_loop is not None:
            logger.info(f'Loop is active. Skipping prediction.')
            return []

        possible_paths = self.find_possible_paths(tracker.events)
        logger.debug(f'Found {len(possible_paths)} possible paths')
        if not possible_paths:
            return []

        search_results: List[Tuple[SearchPath, Iterable[Text]]] = []
        for sp in possible_paths:
            node = sp.nodes[0]
            logger.debug(f'Candidate {node}:')
            possible_next_actions = self._find_possible_next_actions(node, tracker)
            logger.debug(f'Possible next actions: {possible_next_actions}')
            if possible_next_actions:
                search_results.append(
                    (
                        sp,
                        (
                            self._get_action_name(action)
                            for action in possible_next_actions
                        ),
                    )
                )

        next_action_candidates = list(self._calculate_probabilities(search_results))
        logger.debug(f'Next actions candidates: {next_action_candidates}')
        return next_action_candidates


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class StoryGraphPolicy(Policy):
    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
        event_grapth: Optional[Dict[Text, Dict[Text, Any]]] = None,
    ) -> None:
        """Initialize the policy."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)
        self._event_grapth = EventGraph.from_dict(event_grapth or {})

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            'priority': MEMOIZATION_POLICY_PRIORITY,
        }

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> 'StoryGraphPolicy':
        metadata = {}
        try:
            with model_storage.read_from(resource) as directory_path:
                with open(directory_path / cls._metadata_filename(), "r") as file:
                    metadata = json.load(file)
        except ValueError:
            logger.debug(
                f"Couldn't load metadata for policy '{cls.__name__}' as the persisted "
                f"metadata couldn't be loaded."
            )
        return cls(config, model_storage, resource, execution_context, **metadata)

    def presist(self) -> None:
        with self._model_storage.write_to(self._resource) as directory_path:
            with open(directory_path / self._metadata_filename(), "w") as file:
                json.dump(self._metadata(), file)
            viz_path = directory_path / 'story_graph.html'
            self._event_grapth.visualize(str(viz_path))

    def _metadata(self) -> Dict[Text, Any]:
        return {"event_grapth": self._event_grapth.to_dict()}

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "story_graph.json"

    def train(
        self,
        story_graph: StoryGraph,
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        self._event_grapth = EventGraph.from_story_graph(story_graph, domain)
        self.presist()
        return self._resource

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        probabilities = self._default_predictions(domain)

        # print()
        # for i, event in enumerate(tracker.events):
        #     print(i)
        #     print(event.as_dict(), flush=True)

        next_action_candidates = self._event_grapth.find_next_action_candidates(tracker)

        for action_candidate in next_action_candidates:
            action_idx = domain.index_for_action(action_candidate.action_name)
            probabilities[action_idx] = action_candidate.probability

        return self._prediction(probabilities)
