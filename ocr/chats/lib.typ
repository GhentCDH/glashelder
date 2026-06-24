#import "@preview/bulb:0.2.1": dither
#import "@preview/cmarker:0.1.8"

#let bg = oklch(rgb("#e6eff1"))
// #let bg = oklch(rgb("#9FBE83").lighten(90%))
#let fg = bg.darken(70%)
#let dither = false


#let width = 60%

#let user = {
  box(fill: fg, clip: true, width: 24pt, height: 24pt, inset: 0pt, outset: 0pt, radius: 12pt, {
    align(center + horizon, {
      move(
        dy: 15pt,
        {
          circle(
            radius: 6pt,
            fill: fg.lighten(30%).desaturate(30%),
          )
          move(
            dy: -15pt,
            circle(
              radius: 11pt,
              fill: fg.lighten(30%).desaturate(30%),
            ),
          )
        },
      )
    })
  })
}

#let font = (font: "Fraunces", size: 9pt)

#let question(q) = {
  set align(right)
  set text(..font)
  block(width: width, fill: bg.darken(10%), radius: 12pt, inset: 6pt, grid(
    columns: (1fr, 20pt),
    gutter: 15pt,
    grid.cell(align: left, q),
    grid.cell(align: top, user),
  ))
}

#let answer(a) = {
  set text(..font)
  block(width: 50% + width / 3, a)
}

/// - i (str): image path
/// - p (content): prompt with the image
#let img(i, p: none) = {
  set text(..font)
  set align(right)
  let img = if type(i) == content {
    i
  } else if dither {
    image(
      dither(
        read(i, encoding: none),
        size: 800,
        palette: (
          bg,
          fg,
        ),
        contrast: 1.3,
      ),
      height: 300pt,
      scaling: "pixelated",
      fit: "contain",
    )
  } else {
    image(i, width: 100%, height: 300pt, fit: "contain")
  }
  let caption = if type(i) == content {
    "image"
  } else {
    i.split("/").last()
  }
  block(width: width)[
    #figure(
      img,
      caption: caption,
    )

    #if p != none {
      p
    }
  ]
}

/// - data (array): chat data
/// - skip-prompt (bool): skip the textual part of the input prompt
#let chat(data, skip-prompt: false) = {
  let chat = data.at(0).chat.history.messages

  for msg in chat.values().enumerate() {
    let i = msg.at(0)
    let msg = msg.at(1)
    if msg.at("files", default: ()).len() != 0 {
      img("./assets/glas/" + msg.files.at(0).name)
    }
    if msg.role == "user" {
      if not skip-prompt or i != 0 {
        question(msg.content)
      }
    } else {
      answer(cmarker.render(msg.content))
    }
  }
}
