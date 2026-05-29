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

    sed -i ':a;N;$!ba; s/\(.*from [a-z_]*\.cli import [a-z_]*\)/\1\n    from {{name}}.cli import {{name}}/' src/glashelder/__init__.py
    sed -i ':a;N;$!ba; s/\(.*main\.add_command([a-z_]*)\)/\1\n    main.add_command({{name}})/' src/glashelder/__init__.py

jupyter experiment:
    uv run --package {{experiment}} --with jupyter jupyter lab --no-browser

enter:
    docker compose exec --user dev glashelder bash -c 'tmux new-session -d -s main 2>/dev/null; tmux attach -t main'

classify:
    uv run glashelder image-matching batch-classify -w image-matching/wikiart-dino.pkl -g clipcluster/dino.pkl --workers 45 --threshold 0.82 --cache-features --cache-path feature_cache.pkl --cache-save-interval 10 --ml-model model-orb.pkl -o classes.json

generate:
    uv run glashelder clipcluster generate-embeddings --image-dir /mnt/share/shares/glashelder/kikirpa --save-path embeddings/kikirpa-dino.pkl --model dinov2 --count 200000 --batch-size 512

servesearch:
    uv run glashelder embed serve --host 0.0.0.0 --port 5000 --embeddings embeddings/glashelder-clip.pkl
