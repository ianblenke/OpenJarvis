# Channels Module Spec

Multi-platform messaging abstraction for connecting agents to external communication services.

## BaseChannel Protocol (`_stubs.py`)

### REQ-channels.protocol.lifecycle: Channel lifecycle
`BaseChannel` abstract class with `channel_id: str`. `connect()` establishes gateway connection. `disconnect()` closes connection.

### REQ-channels.protocol.messaging: Send/receive messages
`send(channel, content, *, conversation_id="", metadata=None) -> bool` sends a message. `on_message(handler: ChannelHandler)` registers message handler.

### REQ-channels.protocol.status: Connection status
`status() -> ChannelStatus` returns CONNECTED|DISCONNECTED|CONNECTING|ERROR. `list_channels() -> List[str]` returns available channels.

### REQ-channels.protocol.registration: Registry-based registration
All channels use `@ChannelRegistry.register("name")` decorator.

## Message Types

### REQ-channels.message: Channel message format
`ChannelMessage` with `channel`, `sender`, `content`, `message_id`, `conversation_id`, `metadata`.

## Implementations

### REQ-channels.telegram: Telegram bot
### REQ-channels.discord: Discord bot
### REQ-channels.slack: Slack bot
### REQ-channels.whatsapp: WhatsApp Business API
### REQ-channels.whatsapp-baileys: WhatsApp via Baileys (Node.js bridge)
### REQ-channels.webhook: Generic webhook
### REQ-channels.email: Email (SMTP/IMAP)
### REQ-channels.signal: Signal messenger
### REQ-channels.teams: Microsoft Teams
### REQ-channels.matrix: Matrix protocol
### REQ-channels.mattermost: Mattermost
### REQ-channels.irc: IRC
### REQ-channels.feishu: Feishu/Lark
### REQ-channels.bluebubbles: BlueBubbles (iMessage bridge)
### REQ-channels.webchat: Web-based chat
### REQ-channels.google-chat: Google Chat
### REQ-channels.line: LINE Messaging API
### REQ-channels.mastodon: Mastodon social network
### REQ-channels.messenger: Facebook Messenger
### REQ-channels.nostr: Nostr decentralized protocol
### REQ-channels.reddit: Reddit messaging
### REQ-channels.rocketchat: Rocket.Chat
### REQ-channels.twitch: Twitch chat
### REQ-channels.viber: Viber messaging
### REQ-channels.xmpp: XMPP/Jabber protocol
### REQ-channels.zulip: Zulip chat

## Tests

- `tests/channels/test_*.py` - Per-channel adapter tests
