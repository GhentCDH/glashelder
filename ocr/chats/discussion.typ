#import "@preview/cmarker:0.1.8"

#import "lib.typ": answer, img, question, answer, set-theme, chat

// #set page(margin: 10pt, width: auto, height: auto)
#let bg = rgb("#FAF4ED")
#let fg = oklch(25%, 20%, 40deg)

#let bg = oklch(93%, 1.87%, 179.47deg)
#let fg = bg.darken(90%)

// #let bg = white
// #let fg = black
#let heading-font = (font: "Fraunces", variations: (SOFT: 100))
#let normal-font = (font: "IBM Plex Sans")

#set-theme((
  fg: fg,
  bg: bg,
  dither: false,
))

#set page(margin: 15pt, fill: bg)
#set text(fill: fg, ..normal-font)
#show raw: set text(font: "JetBrainsMono NF")
#show heading: set text(..heading-font)
#set heading(numbering: "A.1", supplement: "hfdstk.")
#show raw.where(block: true): set block(fill: bg.darken(5%), inset: 5pt, radius: 12pt)

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
