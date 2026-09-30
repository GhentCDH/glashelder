new name:
    #!/usr/bin/env bash
    uv init --lib {{name}}
    uv add --package {{name}} click
    uv add {{name}} --workspace

    cat << EOF > {{name}}/src/{{name}}/cli.py
    import click

    @click.group()
    def {{name}}():
        pass


    @{{name}}.command()
    @click.option("--option", default="change me", help="change this help message")
    def command(option):
        pass
    EOF

    # register the new group in the lazy COMMANDS table; the help line shows in `--help`
    sed -i 's|^COMMANDS: dict\[str, tuple\[str, str\]\] = {$|&\n    "{{name}}": ("{{name}}.cli:{{name}}", "TODO: describe {{name}}."),|' src/glashelder/__init__.py
    uv run ruff format src/glashelder/__init__.py

jupyter experiment:
    uv run --package {{experiment}} --with jupyter jupyter lab --no-browser

enter:
    docker compose exec --user dev glashelder bash -c 'tmux new-session -d -s main 2>/dev/null; tmux attach -t main'

classify:
    # paths and tuning come from glashelder.toml
    uv run glashelder artmatcher batch-classify -o classes.json --cache-features --cache-save-interval 10

# scrape every backend into paths.artworks (glashelder.toml)
scrape:
    uv run glashelder artscraper scrape

# embed what is new in the scrape into paths.embeddings.reference, fetching the images
reference:
    uv run glashelder embed artworks --batch-size 512

# embed what is new in paths.images.glashelder into paths.embeddings.glashelder
generate:
    uv run glashelder embed images --batch-size 512

servesearch:
    uv run glashelder embed serve --host 0.0.0.0 --port 5000 --embeddings embeddings/glashelder-clip.store
