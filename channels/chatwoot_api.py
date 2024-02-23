from typing import Dict, Any, List
from sanic import Sanic

import logging


logger = logging.getLogger(__name__)


class ChatwootConnector:
    def __init__(self, app: Sanic,
                 api_access_token: str,
                 chatwoot_url: str,
                 account_id: int,
                 conversation_id: int) -> None:
        self.app = app
        self.api_access_token = api_access_token
        self.chatwoot_url = chatwoot_url
        self.account_id = account_id
        self.conversation_id = conversation_id

    async def send_messages(self, messages: List[Dict[str, Any]], is_private: bool = False) -> None:
        """https://www.chatwoot.com/developers/api/#operation/create-a-new-message-in-a-conversation"""
        url = f"{self.chatwoot_url}/api/v1/accounts/{self.account_id}/"\
              f"conversations/{self.conversation_id}/messages"
              
        for message in messages:
            body = {
                'content': message["text"],
                'message_type': "outgoing",
                'private': is_private
            }
            
            async with self.app.aiohttp_session.post(url, json=body) as resp:
                if resp.status != 200:
                    logger.info(f"Response from Chatwoot is not 200:\n"
                                f"code: {await resp.status()},\n"
                                f"body: {await resp.text()}")

    def change_assign(self, agent_id: int) -> None:
        """https://www.chatwoot.com/developers/api/#operation/assign-a-conversation"""
        url = f"{self.chatwoot_url}/api/v1/accounts/{self.account_id}/contacts"
        
        pass

    def update_contact_attrs(self, attrs: Dict[str, Any]) -> None:
        """https://www.chatwoot.com/developers/api/#operation/contactUpdate"""
        pass
