# Règles pour Agents IA (FastAPI-Langchain)

Ces consignes aident un agent IA à contribuer efficacement à ce dépôt. Elles reflètent les pratiques actuelles observées dans le code.

## Vue d’ensemble

- Framework: FastAPI, structure modulaire sous `app/`.
- Séparation claire: routers (HTTP), services (métier), schemas (Pydantic), models (ORM), core (infra), middlewares, errors, templates/static.
- Exemple de référence: `app/main.py` (endpoints démo: path/query/body, `Enum`, `Pydantic`).

## Fichiers et exemples clés

- `app/main.py`: exemples d’endpoints typés, `Enum` (`schemas.enums.ModelName`), payload Pydantic (`ItemCreate`) avec `.model_dump()`.
- `app/schemas/enums.py`: pattern d’énumération (contraindre des params path).
- `app/schemas/pydantic/items.py`: DTO Pydantic avec champs optionnels (`| None = None`).
- `README.md`: architecture globale + diagramme.
- `Makefile`: `make dev` → `uv run fastapi dev app/main.py` (rechargement à chaud).
- `pyproject.toml`: dépendances (`fastapi[standard]`, `python-dotenv`), dev (`ruff`), Python `>=3.14`.

## Workflows de développement

- Démarrer en dev: `make dev` (recommandé). Alternative: `uv run fastapi run app/main.py`.
- Lint: `ruff` (si configuré) via `uv run ruff check .`.
- Tests: dossier `tests/` présent; ajouter des tests `TestClient` FastAPI quand vous créez des endpoints. Runner à ajouter (ex: `pytest`).

## Conventions du projet

- Typage: utiliser Python 3.14 (union `|`), typer explicitement les retours d’endpoints (`-> dict[str, str | int | None]`).
- Sérialisation: Pydantic `BaseModel` et `.model_dump()` pour produire les réponses.
- Organisation des DTO: mettre sous `app/schemas/pydantic/`. Les enums sous `app/schemas/enums.py`.
- Routers vs main: au fur et à mesure, déplacer la logique d’API dans `app/routers/` avec `APIRouter`; garder `app/main.py` pour l’app et l’inclusion des routers.
- Services: logique métier dans `app/services/`, sans dépendances HTTP directes.
- Imports: utiliser des imports absolus ancrés sur `app/` (ex: `from schemas.pydantic.items import ItemCreate`).

## Points d’intégration

- Base de données (futur): config engine/session dans `app/core/`, modèles dans `app/models/`, interactions en services.
- Middlewares: définir sous `app/middlewares/` et enregistrer dans l’app (idéalement via une app factory dans `app/core/`).
- Erreurs: exceptions + handlers centralisés sous `app/errors/` pour un format homogène.
- Templates/Static: Jinja2 sous `app/templates/`, assets sous `app/static/` si rendu HTML.
- Variables d’environnement: utiliser `python-dotenv` + Pydantic Settings (placer la config dans `app/core/`).

## Règles pour agents

- Créer de nouveaux endpoints dans `app/routers/` (préfixes, `tags`) et les inclure dans l’app plutôt que dans `main.py`.
- Réutiliser DTO/enums existants; ajouter les nouveaux sous `app/schemas/`.
- Appeler la logique métier depuis `app/services/`; garder les routers minces.
- Maintenir des annotations de types explicites et des réponses cohérentes avec les exemples.
- Utiliser `make dev` pendant l’itération et ajouter des tests ciblés sous `tests/` pour toute nouvelle route/service.
