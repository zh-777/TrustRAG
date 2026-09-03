# TrustRAG Professional UI Refresh

This is a frontend/UI refresh only. It does **not** create a new Axiom model generation and does not replace the RAG-1 backend architecture.

## Main UI changes
- Clean light professional workspace inspired by the supplied reference.
- Collapsible navigation sidebar with New Chat, Knowledge Bases, recent conversations, API key status, and account access.
- Full-height conversation canvas with independent scroll behavior.
- GroundCheck source/verification rail on the right with Sources and Details tabs.
- Fixed bottom composer that remains fully visible at all viewport heights.
- Axiom mode selector available from the top bar and composer; all existing modes remain connected to the same `axiom_mode` request field.
- File attachments appear in the composer and in the sent user message.
- Knowledge settings moved into a dedicated modal without removing Dataset/Text/File logic or retrieval depth.
- Provider/API-key/model settings moved into a secure dedicated modal. API keys remain masked and session-only.
- The Axiom SVG is static during idle use and animates only while generation is active.
- Smooth Framer Motion transitions and responsive desktop/tablet/mobile layout.

## Logic preserved
- `/api/ask`
- `/api/media/extract`
- Dataset row loading
- General / Grounded / Hybrid / Auto / Local modes
- Provider/BYOK model selection
- GroundCheck result data
- Authentication and chat persistence
- Uploaded source extraction and attachment state

The four "Model Utilities" items in the reference UI are displayed as disabled future-capability placeholders so the frontend does not pretend functionality exists when the backend does not currently provide it.
