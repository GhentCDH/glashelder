// Includebaar fragment: leg de sleutels van het validatierapport uit.
// Bevat geen preamble; wordt via `#include` opgenomen in discussion.typ.

= Het validatierapport <metrics>

De validatie vergelijkt voor elke afbeelding de #emph[ground truth] (handmatige
annotatie) met de #emph[predictie] van het model. De vergelijking is
*eenrichting*: we zoeken elk woord uit de ground truth terug in de predictie,
niet omgekeerd. De ground truth bevat enkel de woorden die ertoe doen; overige
tekst in de afbeelding hoeft niet getranscribeerd te zijn.

Matching gebeurt *woord per woord* met een genormaliseerde Levenshtein-afstand,
en blijft *binnen de sectie* (`meta` of `content`) waar het ground-truth-woord
staat. Een woord dat enkel in de andere sectie wordt teruggevonden, telt als
#emph[misplaced] — zo wordt categorie-lekkage zichtbaar.

Het rapport heeft drie toplagen: `config`, `overall` en `images`.

== `config`

Vastlegging van hoe de run gescoord werd, zodat het rapport zelf-documenterend is.

/ `tau`: de gelijkenisdrempel $tau in [0,1]$. Een ground-truth-woord telt als
  teruggevonden wanneer de beste gelijkenis $>= tau$. Standaard `0.8`.
/ `tokenizer`: hoe tekst in woorden wordt opgesplitst — accenten gestript
  (é → e), kleine letters, en `\w+` (splitst dus ook op leestekens).
/ `note`: herinnering dat de score eenrichting is (GT → predictie) en per
  sectie begrensd.

== `overall`

De globaal samengevoegde cijfers, #emph[micro-averaged]: alle woorden van alle
afbeeldingen worden op één hoop gegooid en dan gescoord (grote afbeeldingen
wegen dus zwaarder dan bij een macro-gemiddelde). Drie ingangen, elk een
#emph[SectionMetrics]-blok (zie hieronder):

/ `meta`: enkel woorden uit de meta-secties.
/ `content`: enkel woorden uit de content-secties.
/ `combined`: meta en content samen.

== `images`

Per afbeelding (sleutel = bestandsnaam zonder extensie) een blok met twee
secties, `meta` en `content`. Elke sectie bevat:

/ `metrics`: een #emph[SectionMetrics]-blok voor die sectie van die afbeelding.
/ `words`: de lijst met losse woordresultaten (#emph[WordResult]), handig om
  individuele fouten na te kijken.

== SectionMetrics — de eigenlijke metrieken

Deze sleutels komen terug in `overall.*` en in `images.*.{meta,content}.metrics`.
Noemer overal is het aantal ground-truth-woorden in die sectie.

/ `n_gt_words`: aantal ground-truth-woorden — de noemer.
/ `recall_at_tau`: $"found" \/ "n_gt"$. Fractie ground-truth-woorden die
  teruggevonden zijn in de #emph[juiste] sectie, boven drempel $tau$. De
  hoofdmaat voor detectie.
/ `recall_any_section`: $("found" + "misplaced") \/ "n_gt"$. Zelfde, maar negeert
  in welke sectie het woord stond — het plafond als categorie-plaatsing er niet
  toe deed.
/ `mean_best_sim`: gemiddelde van de beste in-sectie-gelijkenis over #emph[alle]
  ground-truth-woorden (niet-gevonden woorden tellen mee met hun lage score).
  Een zachte score in $[0,1]$ zonder drempelkeuze.
/ `cer`: #emph[character error rate] over de gevonden paren — som van de
  Levenshtein-afstanden gedeeld door de totale ground-truth-lengte. Meet de
  transcriptie-#emph[kwaliteit] los van de detectie. Enkel berekend over
  woorden met status `found`.
/ `misplaced_rate`: fractie ground-truth-woorden die enkel in de #emph[verkeerde]
  sectie teruggevonden zijn — de categorie-lekkage.
/ `missed_rate`: fractie ground-truth-woorden die #emph[nergens] (boven $tau$)
  teruggevonden zijn.
/ `n_found`, `n_misplaced`, `n_missed`: de absolute tellingen achter de
  bovenstaande fracties. Het geldt dat
  $"n_found" + "n_misplaced" + "n_missed" = "n_gt_words"$.

Er is bewust *geen precision*: de validatie is eenrichting. Extra woorden in de
predictie die niet in de ground truth staan, worden niet afgestraft — ze hoeven
daar niet te zijn.

== WordResult — per woord

Elk element van `images.*.{meta,content}.words`:

/ `gt`: het genormaliseerde ground-truth-woord.
/ `match`: de best passende kandidaat uit #emph[dezelfde] sectie van de
  predictie (of `null` als die sectie leeg was).
/ `sim`: de gelijkenis van dat beste in-sectie-paar, in $[0,1]$.
/ `sim_other`: de beste gelijkenis in de #emph[andere] sectie, in $[0,1]$.
  Wordt opgeslagen zodat een rapport bij een andere `tau` herrekend kan worden
  (`ocr rescore`) zonder hertranscriptie.
/ `status`: een van
  - `found` — `sim` $>= tau$ in de juiste sectie;
  - `misplaced` — onder $tau$ in de eigen sectie, maar $>= tau$ in de andere;
  - `missed` — nergens boven $tau$.
