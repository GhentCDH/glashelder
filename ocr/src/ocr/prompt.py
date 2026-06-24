# the only allowed values for an entry's "type"; mirrored in cli.OUTPUT_SCHEMA
TYPES = ("text", "creator", "place", "year")

_TYPE_GUIDE = """Each transcribed entry has a "text" value and a "type". Choose the type that best fits:
- "creator": a name of a creator (e.g. the maker of an artwork)
- "place": everything related to places (e.g. a street sign)
- "year": a year (e.g. 1867)
- "text": anything else / general text not covered above

Always split a passage into separate entries when part of it can carry a more specific type than "text". Never leave a place, creator or year buried inside a "text" entry if it can be isolated.
For example "BEIRLEGHEM. Oud vrouwtje." must become two entries:
{"text": "BEIRLEGHEM", "type": "place"}
{"text": "Oud vrouwtje.", "type": "text"}
"""

PROMPT = f"""
You are an expert transcription assistant. Your task is to extract ALL readable text from this image of a scanned glass plate.

Glass plates have two distinct text zones:
1. **Border/margin area** (surrounding the photograph): labels, titles, identifiers, and annotations added by the photographer or archivist. These go in "meta".
2. **Within the photographic scene itself**: text that was physically present in the scene when the photo was taken (e.g. signs, banners, text on objects). These go in "content".

If no text appears within the photographic scene, return an empty array for "content".

{_TYPE_GUIDE}
Output structured JSON and nothing else. Example structure:
{{
  "meta": [{{"text": "Gent", "type": "place"}}],
  "content": [{{"text": "1867", "type": "year"}}]
}}
"""

CONTENT_ONLY_PROMPT = f"""
You are an expert transcription assistant. This image is the photographic scene of a glass plate, with no border/margin annotations.

Extract ALL text that was physically present in the scene (signs, banners, text on objects). These go in "content". Leave "meta" empty.

{_TYPE_GUIDE}
Output structured JSON and nothing else. Example structure:
{{
  "meta": [],
  "content": [{{"text": "1867", "type": "year"}}]
}}
"""

META_ONLY_PROMPT = f"""
You are an expert transcription assistant. This is a full glass plate scan with the photographic scene blurred out.

Read ONLY the border/margin labels, titles, identifiers and annotations. These go in "meta". Leave "content" empty.

{_TYPE_GUIDE}
Output structured JSON and nothing else. Example structure:
{{
  "meta": [{{"text": "Gent", "type": "place"}}],
  "content": []
}}
"""

CROP_CONTENT_PROMPT = f"""
You are an expert transcription assistant. This is a crop of the photographic scene of a glass plate, with no border/margin area.

Extract ALL text physically present in the scene (signs, banners, text on objects). These go in "content". Leave "meta" empty.

{_TYPE_GUIDE}
Output structured JSON and nothing else. Example structure:
{{
  "meta": [],
  "content": [{{"text": "1867", "type": "year"}}]
}}
"""

TWO_IMAGE_PROMPT = f"""
You are an expert transcription assistant transcribing a scanned glass plate, given TWO images:

1. The FIRST image is the full plate with the photographic scene blurred out. Read only the border/margin labels, titles, identifiers and annotations here. These go in "meta".
2. The SECOND image is the sharp crop of the photographic scene. Read only the text physically present in the scene (signs, banners, text on objects). These go in "content".

Do not transcribe blurred/unreadable text. If a zone has no text, return an empty array for it.

{_TYPE_GUIDE}
Output structured JSON and nothing else. Example structure:
{{
  "meta": [{{"text": "Gent", "type": "place"}}],
  "content": [{{"text": "1867", "type": "year"}}]
}}
"""
