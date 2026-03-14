# Sessions Module Spec

Cross-channel persistent session management with identity linking and message consolidation.

## Session Types

### REQ-sessions.identity: Session identity
`SessionIdentity` with `user_id`, `display_name`, `channel_ids: Dict[str, str]` (channel_type -> channel_user_id).

### REQ-sessions.session: Session object
`Session` with `session_id`, `identity`, `messages`, `created_at`, `last_activity`, `metadata`.

## SessionStore

### REQ-sessions.store.get-or-create: Session resolution
`get_or_create(user_id, *, channel, channel_user_id, display_name) -> Session` finds existing or creates new session.

### REQ-sessions.store.messages: Message persistence
`save_message(session_id, role, content, *, channel, metadata)` persists messages.

### REQ-sessions.store.consolidation: Message consolidation
`consolidate(session_id)` summarizes old messages to reduce context size.

### REQ-sessions.store.decay: Session cleanup
`decay(max_age_hours) -> int` removes expired sessions.

### REQ-sessions.store.channel-linking: Cross-channel linking
`link_channel(session_id, channel, channel_user_id)` links additional channel identities.

## Session Types (detailed)

### REQ-sessions.identity.create: Identity creation
SessionIdentity can be created with user_id, display_name, and channel_ids mapping.

### REQ-sessions.identity.defaults: Identity default values
SessionIdentity provides defaults for optional fields (empty channel_ids, None display_name).

### REQ-sessions.message.create: Message creation
Session messages are created with role, content, timestamp, and optional channel.

### REQ-sessions.message.metadata: Message metadata
Session messages support arbitrary metadata dict for channel-specific or application-specific data.

### REQ-sessions.session.create: Session creation
Sessions are created with auto-generated ID, identity, empty message list, and timestamps.

### REQ-sessions.session.add-message: Session message addition
`Session.add_message()` appends messages and updates the last_activity timestamp.

## SessionStore (detailed)

### REQ-sessions.store.create: Session store creation
SessionStore creates new sessions with identity and optional channel linking.

### REQ-sessions.store.get-existing: Get existing session
SessionStore returns the same session when queried with matching user_id/channel.

### REQ-sessions.store.save-message: Message persistence
SessionStore persists messages and reloads them with the session.

### REQ-sessions.store.cross-channel: Cross-channel session linking
SessionStore links multiple channel identities to a single session for unified conversation.

### REQ-sessions.store.link-channel: Channel linking
SessionStore links additional channel identities to an existing session.

### REQ-sessions.store.list: Session listing
SessionStore lists all sessions with optional filtering.

### REQ-sessions.store.expiry: Session expiry
SessionStore supports session expiry based on last_activity timestamp.

### REQ-sessions.store.expiry-with-channel: Session expiry with channel context
SessionStore creates a new session with channel linking when the previous session has expired.

### REQ-sessions.store.get-existing-with-channel: Get existing session with channel update
SessionStore updates channel_ids on an existing session when queried with new channel information.

## Tests

- `tests/sessions/test_sessions.py` - Session management tests
