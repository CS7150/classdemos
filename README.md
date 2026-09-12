# classdemos

Demos for CS 7150 Deep Learning at Northeastern University.

**Site: https://cs7150.github.io/classdemos/**

The site indexes every notebook in this repository (each with a one-click Colab
link) and hosts the browser-based interactive demos under `docs/demos/`.

## Repository layout

- `*.ipynb` — lecture and homework notebooks
- `cs7150/` — small helper package a few notebooks import
- `docs/` — the GitHub Pages site (`index.html` plus `demos/`)

## Publishing the site

GitHub Pages serves from the `main` branch, `/docs` folder
(Settings → Pages → Source: "Deploy from a branch" → `main` / `/docs`).
Adding a demo means dropping a self-contained `.html` file into `docs/demos/`
and adding a card for it in `docs/index.html`.
