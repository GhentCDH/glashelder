PROMPT = """
You are an expert transcription assistant. Your task is to extract ALL readable text from this image of a scanned glass plate.

Glass plates have two distinct text zones:
1. **Border/margin area** (surrounding the photograph): labels, titles, identifiers, and annotations added by the photographer or archivist. These go in "meta".
2. **Within the photographic scene itself**: text that was physically present in the scene when the photo was taken (e.g. signs, banners, text on objects). These go in "content".

If no text appears within the photographic scene, return an empty array for "content".

Output structured JSON and nothing else. Example structure:
{
  "meta": [{"text": "..."}],
  "content": [{"text": "..."}]
}
"""

CONTENT_ONLY_PROMPT = """
You are an expert transcription assistant. This image is the photographic scene of a glass plate, with no border/margin annotations.

Extract ALL text that was physically present in the scene (signs, banners, text on objects). These go in "content". Leave "meta" empty.

Output structured JSON and nothing else. Example structure:
{
  "meta": [],
  "content": [{"text": "..."}]
}
"""

TWO_IMAGE_PROMPT = """
You are an expert transcription assistant transcribing a scanned glass plate, given TWO images:

1. The FIRST image is the full plate with the photographic scene blurred out. Read only the border/margin labels, titles, identifiers and annotations here. These go in "meta".
2. The SECOND image is the sharp crop of the photographic scene. Read only the text physically present in the scene (signs, banners, text on objects). These go in "content".

Do not transcribe blurred/unreadable text. If a zone has no text, return an empty array for it.

Output structured JSON and nothing else. Example structure:
{
  "meta": [{"text": "..."}],
  "content": [{"text": "..."}]
}
"""
