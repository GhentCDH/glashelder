#import "@local/templater:2.0.0": quick
#import "@preview/cetz:0.5.2"
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
it's important that each `Artwork` has a `source` (e.g. ```py "wikiart"```), a `url` (e.g. ```py "https://example.com/artworkX"```) and an `image_url` (e.g. #link("https://cataas.com/cat")[```py "https://cataas.com/cat"```]):

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

we also set ```zsh --limit 10``` because otherwise, this test run would potentially download multiple thousands of artworks, which is not needed for a simple test!

== letting llms help you

once you've found out which data source you want to scrape, you can write out a prompt for an llm, asking it to help you with the implementation.

send this `.pdf` document along with it, and also add `./src/artscraper/backends/__init__.py` for `Backend` and `Context` and `./src/artscraper/backends/kikirpa.py` as an example.

== fyi

don't be alarmed when you see some new concepts like ```py await``` or ```py AsyncIterator```

most of the work a scraper does is *waiting*. when you ask a website for a page, your program sends the request and then just sits there, doing nothing, until the answer comes back over the network. that waiting is the slow part, not the code.

if we did this the normal way, one page after another (also called 'sequentially'), every request would have to finish before the next one starts. all that waiting stacks up:

#let clr = rgb("#26657e")
#let clr2 = rgb("#538E99")

#{
  set text(font: "Maple Mono")
  align(center, cetz.canvas({
    import cetz.draw: *

    let bar(x, y, w, label) = {
      rect((x, y), (x + w, y - 0.6), fill: clr, stroke: clr2)
      content((x + w / 2, y - 0.3), text(size: 6pt, fill: white, label))
    }

    // sequential: each request waits for the previous one
    content((-0.2, 0), text(size: 8pt, weight: "bold")[sequential], anchor: "east")
    bar(0, 0, 2.2, "wait for page 1")
    bar(2.2, 0, 2.2, "wait for page 2")
    bar(4.4, 0, 2.2, "wait for page 3")

    // async: all requests wait at the same time
    content((-0.2, -1.4), text(size: 8pt, weight: "bold")[async], anchor: "east")
    bar(0, -1.4, 2.2, "wait for page 1")
    bar(0.1, -2.1, 2.2, "wait for page 2")
    bar(0.2, -2.8, 2.2, "wait for page 3")

    line((2.4, 0.6), (2.4, -3.55), stroke: (dash: "dotted", paint: gray))
    line((6.6, 0.6), (6.6, -3.55), stroke: (dash: "dotted", paint: gray))
    content((6.6, -3.7), text(size: 7pt, fill: gray)[sequential finishes here])
    content((2.2, -3.7), text(size: 7pt, fill: gray)[async finishes here])
  }))
}

that's the whole idea behind ```py async``` and ```py await```. while one request is waiting for its answer, python is free to fire off the next request instead of just waiting. the ```py await``` keyword marks the spots where waiting happens, so python knows it can go do other useful work there in the meantime.

```py AsyncIterator``` is just the type of thing your ```py stream``` function produces: a sequence of artworks that arrives a bit at a time (as pages come back) rather than all at once. that's why you ```py yield``` each artwork instead of building one big list and returning it. ```py yield```#text(font: "Maple Mono", size: 9pt)[ing] means 'i have some more data for you!' but lets the function continue again afterwards.

you don't need to fully understand the machinery. the takeaway: writing ```py async def``` and ```py await fetch_text(...)``` lets the scraper wait for many pages at the same time, which makes it dramatically faster than fetching them one by one.
