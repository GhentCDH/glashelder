#import "@local/templater:2.0.0": quick
#show: quick.with()
#set document(title: "Artscraper Help", author: "Miel Peeters")

#set heading(numbering: (..nums) => {
  let vals = nums.pos()
  if vals.len() == 3 {
    numbering("1.", vals.last())
  }
})

#show heading.where(level: 3): set text(size: 10pt, style: "italic")

= artscraper help

#box(
  stroke: (left: (thickness: 1pt, cap: "round")),
  outset: (top: 0.2em, bottom: 0.2em),
  inset: (left: 0.8em),
)[
  context: glashelder needs a big dataset of reference artworks, which we 'scrape' from public websites.
]

== a harness uses backends

there is a harness or orchestrator which runs one or more 'backends' side by side.
all this harness essentially does is this:

```py
# harness.py
backends = [...]
context = Context(...)

artworks = []
for backend in backends:
  for artwork in backend.stream(context):
    artworks.append(artwork)

write_to_json(artworks)
```

#place(right, dy: -2em, align(right, text(size: 8pt)[(it's more complicated but this is the gist of it)]))

a 'backend' is thus one source of artworks.\
to add a new datasource, we need to write a new ```py class NewDataSource(Backend)```, which also has the ```py Backend.stream(context)``` function.

== backends

backends live in `./src/artscraper/backends/`.

the `__init__.py` file there has some helper functions, which you don't need to really care about, and defines what a Backend is:

```py
class Backend(Protocol):
    name: str

    def stream(self, ctx: Context) -> AsyncIterator[Artwork]: ...
```

that ```py Context()``` object contains some things that your backend needs to use:

- ```py Context.client```: this is a client you can use to get webpages in your script, for example: \
  ```py
  # import a helper function that gives the html page at `url` as a str
  from artscraper.client import fetch_text

  html = await fetch_text(ctx.client, "https://example.com/", ctx.throttle, ctx.stats)
  return parse_page(url, html)
  ```
- the rest you don't need to worry about `:)`

you can see that the `Backend` produces `Artwork`s, which bundle some information about an artwork that we found.
it's important that each `Artwork` has a `source` (e.g. ```py "wikiart"```), a `url` (e.g. ```py "https://example.com/artworkX"```) and a `image_url` (e.g. #link("https://cataas.com/cat")[```py "https://cataas.com/cat"```]):

```py
# ... hard work to get to a place where you found all of the necessary information
artwork = Artwork(
  source="cute_cats",
  url="https://example.com/cat",
  image_url="https://cataas.com/cat"
)
```

let's walk through an example; to add a new `Backend` for the imaginary database called 'Cute Cat Artworks', you need to:

=== create a new file

`./src/artscraper/backends/cute_cats.py`

=== implement the new `Backend` class

```py
from ..model import Artwork
from . import Backend, Context

def find_image_url(html: str) -> str | None:
  ...

def parse_artwork_page(url: str, html: str) -> Artwork | None:
  """Create an ``Artwork`` from the page's html"""

  image_url = find_image_url(html)  # your own parsing
  if image_url is None:
    return None  # no image, not a usable artwork

  return Artwork(
    source="cutecats",
    url=url,
    image_url=image_url,
  )

class CuteCats(Backend):
  name = "cutecats"

  async def stream(self, ctx: Context) -> AsyncIterator[Artwork]:
    for url in ["https://cataas.com/cat/orange,cute", "https://cataas.com/cat/white"]:
      html = await fetch_text(ctx.client, url, ctx.throttle, ctx.stats)
      artwork = parse_artwork_page(url, html)
      if artwork is not None:
        yield artwork
```

=== wire up the harness

list the newly created `Backend` in the registry (at the bottom of `./src/artscraper/backends/__init__.py`)

```py
 def registry() -> dict[str, Backend]:
     """All configured backends by name."""
+    from .cute_cats import CuteCats
     from .kikirpa import Kikirpa
     from .wikiart import WikiArt

-    return {backend.name: backend for backend in (WikiArt(), Kikirpa())}
+    return {backend.name: backend for backend in (WikiArt(), Kikirpa(), CuteCats())}
```

== testing your scraper

to test the new backend, we can run the `scrape` command with ```bash --backend cutecats```:

```zsh
uv run glashelder artscraper scrape --backend cutecats --limit 10
```

we also set ```zsh --limit 10``` because otherwise, this test run would potentially download multiple thousands of artworks, which is not needed for a simple test!.

== fyi

don't be alarmed when you see some new concepts like ```py await``` or ```py AsyncIterator```

// claude: here i want a very short overview of why we add these things, and what async means.
// it'd be nice if you could draw a diagram of some sort of waterfall like timing thing comparing async io request with sequential io bound things (like horizontal rectangles, stacked, representing requests. sequential puts them after each other, showing that it takes a long time, async below each other showing parallellism and)
//
// please give an overview of what 'io waiting' is, sort of. but please don't introduce new vocabulary that they don't need
// this is intended for students that are working on this task, they don't need to get a complex description, just
// a clearer understanding of what is happening and why!
