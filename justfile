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

    sed -i '/from.*cli import/a\    from {{name}}.cli import {{name}}' src/glashelder/__init__.py
    sed -i '/main.add_command/a\    main.add_command({{name}})' src/glashelder/__init__.py

jupyter experiment:
    uv run --package {{experiment}} --with jupyter jupyter lab --no-browser

enter:
    docker compose exec --user dev glashelder bash -c 'tmux new-session -d -s main 2>/dev/null; tmux attach -t main'
