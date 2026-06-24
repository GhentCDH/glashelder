#import "@preview/cmarker:0.1.8"
#import "@preview/bulb:0.2.1": dither
#import "@preview/based:0.2.0": base64

#import "lib.typ": answer, bg, chat, fg, img, question

// #set page(margin: 10pt, width: auto, height: auto)
#let heading-font = (font: "Fraunces", variations: (SOFT: 100))
#let normal-font = (font: "IBM Plex Sans")

#set page(margin: 15pt, fill: bg)
#set text(fill: fg, ..normal-font)
#show raw: set text(font: "JetBrainsMono NF")
#show heading: set text(..heading-font)
#set heading(numbering: "A.1", supplement: "hfdstk.")
#show raw.where(block: true): set block(fill: bg.darken(5%), inset: 5pt, radius: 12pt)
#show raw.where(block: true): set align(center)

#let title = [Glashelder: OCR Experiments log]

#set document(
  author: "Miel Peeters <miel.peeters@ugent.be>",
  description: [Log of experiments performed on OCRing Glashelder images],
  title: title,
)

#text(size: 30pt, weight: 600, ..heading-font)[#title]

Dit document beschrijft de ontdekkende experimenten van verschillende OCR-approaches die plaatsvonden
tijdens de voorbereidende fase van sprint 4 van Glashelder.

= VLM OCR: Qwen3.5

Meer specifiek: `Qwen/Qwen3.5-35B-A3B`, via VLLM container op DGX Spark.

Het model werd als volgend opgestart:

#raw(read("qwen35-command.sh"), lang: "sh", block: true)

In dit experiment werden een aantal _niet-gecropte_ glashelder afbeeldingen aan de VLM gegeven met de volgende prompt:

#raw(read("prompt0.md"), lang: "md", block: true)

== Vlaskeerders <vlas1>

#chat(json("./chats/vlaskeerders_00.json"))

== Richtingaanwijzingen

#chat(json("./chats/richting_00.json"))

== Muziek

#chat(json("./chats/music_00.json"))

#divider()

Het is duidelijk dat het model genoeg intelligentie heeft om zowel de tekst in de foto zelf als de labels op de glasplaat
te lezen.
De mate waarin het model _'moeite doet'_ om te _alle_ tekst te lezen en te beschrijven waar ze vandaan komt varieert nog wel sterk.

Bijvoorbeeld de foto van de vlaskeerders @vlas1: de ronde sticker rechts onderaan werd genegeerd. \
Wanneer dezelfde foto met dezelfde prompt nog eens werd gegeven, transcribeerde het model de sticker wel...

De prompt werd bijgestuurd naar dit:

#block(fill: bg.darken(5%), inset: 5pt, radius: 12pt, raw(read("prompt1.md"), lang: "md"))

En dezelfde afbeeldingen werden nogmaals getoond aan het model:

== Vlaskeerders

#chat(json("./chats/vlaskeerders_01.json"))

== Richtingaanwijzingen

#chat(json("./chats/richting_01.json"))

== Muziek

#chat(json("./chats/music_01.json"))

#divider()

Dit maakt duidelijk dat het mogelijk is om het model te sturen om wel degelijk alle leesbare tekst the transcriberen.

= VLM OCR: Gemma4

Meer specifiek: `google/gemma-4-26B-A4B-it`, via VLLM container op DGX Spark.

Het model werd als volgend opgestart:

#raw(read("gemma-command.sh"), lang: "sh", block: true)

De laatste, meer uitgewerkte, prompt werd gebruikt voor de volgende tests:

== Vlaskeerders

#chat(json("./chats/vlaskeerders_02.json"), skip-prompt: true)

== Richtingaanwijzingen

#chat(json("./chats/richting_02.json"), skip-prompt: true)

== Muziek

#chat(json("./chats/music_02.json"), skip-prompt: true)

#divider()

Dit model lijkt heel gelijkaardig te presteren aan qwen3 voor deze tests.


De belangrijkste conclusie is dat de modellen er in slagen om relevante tekst uit afbeeldingen te lezen, over zeer verschillende inhoud heen.
De keuze tussen de beschikbare modellen lijkt mij minder interessant om veel tijd aan te spenderen.

De volgende tests gaan verder met `Qwen3.5`.

= Structured Output (`json`)

In dit experiment beginnen we met automatisatie van het bevragen van
de VLM modellen.

Geen chat-interface meer, maar API requests met de OpenAPI-compatibele
interface van VLLM. Op die manier kunnen we zelf aangeven in welk formaat we verwachten dat de LLM antwoordt, en welke velden er in moeten staan.

In de vorige experimenten vroegen we gewoon om alle tekst te geven die in de afbeelding staat, maar het model geeft niet duidelijk en eenduidig aan waar ze deze vandaan heeft gehaald.
Voor glashelder is het belangrijk dat we weten of de tekst deel was van de foto zelf, of van de annotaties op de glasplaat.

Nu kunnen we opleggen dat het model haar antwoorden opsplitst in `content` en `meta` tekst.

Het maakt het ook veel eenvoudiger om grotere sets van afbeeldingen te verwerken:

#align(center, block(width: 70%, figure(
  grid(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    rows: 50pt,
    gutter: 10pt,
    ..toml("./images.toml")
      .paths
      .map(i => image(
        i,
        width: 100%,
        height: 50pt,
        fit: "contain",
      ))
  ),
  caption: [een reeks afbeeldingen die aan `Qwen 3.5` werd gegeven],
)))

Hier is de `json` structuur die we opleggen aan de antwoorden van het model:

```py
_TEXT_ENTRY = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "meta": {"type": "array", "items": _TEXT_ENTRY},
        "content": {"type": "array", "items": _TEXT_ENTRY},
    },
    "required": ["meta", "content"],
    "additionalProperties": False,
}
```

This means that we'll get responses that look like this:

```json
{
  "meta": [
    {"text": "BW 41254"}
  ],
  "content": [
    {"text": "Groeten uit"},
    {"text": "BRUSSEGEM"}
  ],
}
```

== Één prompt

De bovenstaande reeks afbeeldingen werden gegeven aan `Qwen 3.5` met de volgende prompt:

#raw(read("prompt2.md"), lang: "md", block: true)

Deze aanpak bleek al snel te simpel.
Het model slaagt er in de meeste gevallen in om de tekst op de labels te onderscheiden van
tekst op de foto zelf, maar in situaties waar de foto zelf eerder 'grafisch' is maakt het fouten.

Bijvoorbeeld:

#img("./assets/glas/0000032h8q_tiff.jpg")
#question("... prompt ...")
#show raw.where(block: true): set align(left)
#raw(read("./assets/jsons-one-prompt/0000032h8q_tiff.json"), lang: "json", block: true)
#show raw.where(block: true): set align(center)

De tekst 'Groeten uit BRUSSEGEM' is juist getranscribeerd, maar staat effectief in het bronmateriaal zelf,
terwijl `Qwen` het bij `meta` zet. Waarschijnlijk is het in de war omdat de tekst lijkt alsof het geschreven is op stickers
die op de foto geplakt zijn, net zoals bij de annotaties van de glasplaat vaak het geval is.

#pagebreak(weak: true)
== Conditionele Prompts

We hebben al gecropte versies van de afbeeldingen en we weten de bounding boxes van die crops.

Daarmee kunnen we een opsplitsing maken:

1. De hele glasplaatscan is inhoudelijk (aka `crop == image`)
  - geef de afbeelding as-is met een prompt die aanduidt dat all tekst in `content` moet komen
2. De glasplaatscan heeft een inhoudelijke crop en randen
  - geef twee afbeeldingen:
    + crop (die dus enkel inhoudt bevat)
    + volledige scan, met blurry inhoud
  - en bijhorende prompts

Bijvoorbeeld:

#let _images(data) = {
  let img1 = data.input.at(1).image_url.url
  let b64 = img1.split(",").last()
  let img-data1 = bytes(base64.decode(b64))
  let img2 = data.input.at(2).image_url.url
  let b64 = img2.split(",").last()
  let img-data2 = bytes(base64.decode(b64))
  (img-data1, img-data2)
}

#let data = json("./assets/jsons-conditional/0000032r8n_tiff.json")
#let (im1, im2) = _images(data)
#img(image(im1, format: "jpg", height: 200pt))
#img(image(im2, format: "jpg", height: 200pt))
#{ data.output.meta = data.output.meta.slice(0, count: 4) }
#{ data.output.content = data.output.content.slice(0, count: 4) }
#answer(json.encode(data.output))

Het antwoord van het model werd hier ingekort voor de duidelijkheid.

Met deze toevoeging slaagt het model er steeds in de gevonden tekst in het juiste veld te plaatsen.

#divider()

Hier is een voorbeeld waarbij deze aanpak natuurlijk faalt: wanneer de preprocessing een foute bounding box tekent.
Dat introduceert meerdere problemen voor de OCR stap:

+ tekst kan worden 'afgekapt' wanneer de foute boundig box deze overtekent
+ juist getrascribeerde tekst wordt mogelijks bij de foute categorie geplaatst ('meta' #sym.arrow.l.r.long 'content')

#let data = json("./assets/misplaced.json")
#let (im1, im2) = _images(data)
#img(image(im1, format: "jpg", height: 200pt))
#img(image(im2, format: "jpg", height: 200pt))
#{ data.output.meta = data.output.meta.slice(0, count: 4) }
#{ data.output.content = data.output.content.slice(0, count: 1) }
#answer(json.encode(data.output))

= Updated JSON structure

We hebben het output formaat aangepast zodat elke entry ook een `"type"` veld heeft. Dat wordt als volgt gedefineerd:

```md
Each transcribed entry has a "text" value and a "type". Choose the type that best fits:
- "creator": a name of a creator (e.g. the maker of an artwork)
- "place": everything related to places (e.g. a street sign)
- "year": a year (e.g. 1867)
- "text": anything else / general text not covered above

Always split a passage into separate entries when part of it can carry a more specific type than "text". Never leave a place, creator or year buried inside a "text" entry if it can be isolated.
For example "BEIRLEGHEM. Oud vrouwtje." must become two entries:
{"text": "BEIRLEGHEM", "type": "place"}
{"text": "Oud vrouwtje.", "type": "text"}
```

Het doel van deze aanpassing is om later makkelijker links
te maken met de geOCRde tekst en bijvoorbeeld geodata.
In het voorbeeld van de prompt zelf zou het bijvoorbeeld triviaal zijn om 'Beirleghem' te vinden als een goed gedefineerde plaats, omdat het model het al heeft gelabelt als een `"place"`.

Daarbij merken we dat deze aanpassing een aanzienlijke verlaging van de _'character error rate'_ met zich mee bracht.
Zie @comparison.

Anderzijds werd de `miss_rate` wel hoger. We hebben nog niet concreet gevonden waaraan dit kan liggen.
Zie @metrics voor een beschrijving van deze metrieken.

#include "validation-report.typ"

#include "report.typ"
