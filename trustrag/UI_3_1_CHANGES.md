# Axiom UI 3.1 — Interface & GroundCheck fixes

## Security / API key UX
- API keys are no longer kept visible in the input after entry.
- Pasting a key saves it immediately for the current browser tab and clears the field.
- Manually typed keys can be saved with the Save control or Enter.
- Saved keys show only a "Saved" state and can be cleared explicitly.
- Keys use `sessionStorage`, not persistent application storage.

## GroundCheck correctness
- Fixed RoBERTa-MNLI usage so premise and hypothesis are passed as a real sentence pair instead of a manually concatenated string.
- Added direct-source support detection for quotations, OCR text, labels and headings.
- Improved source segmentation for line-based / OCR text.
- List items are preserved as complete claims rather than being fragmented into tiny sentences.
- Contradiction verdicts now require stronger evidence, reducing false contradictions.
- UI now calls the percentage "faithful" instead of implying that selecting Grounded mode guarantees a score.

## Conversation layout
- Collapsing the sidebar expands the workspace to the full viewport.
- The conversation owns the full-width scroll area so the scrollbar stays on the far right.
- The composer is now outside the scrolling container and fixed safely above the bottom edge, preventing clipping.
- Axiom mode selection is located in the conversation header (Auto / Grounded / Hybrid / General / Local).
- Removed the duplicate mode selector from the composer.

## Upload experience
- Uploaded knowledge appears immediately in the composer as an attachment card.
- Image uploads get a thumbnail preview.
- File/audio/video uploads get type-aware attachment cards.
- The attachment is shown with the user's message after sending.

## Motion / branding
- The supplied Axiom SVG itself is now static.
- Axiom only pulses/animates while a response is being generated.
- Added subtle Framer Motion transitions and hover/focus motion for panels, messages, attachments and menus.
- Added reduced-motion support.
