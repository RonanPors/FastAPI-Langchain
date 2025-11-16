from fastapi import FastAPI
from schemas.pydantic.items import ItemCreate

app = FastAPI()


# Déclarer un endpoint sans params
@app.get(
    "/",
    summary="Accueil",
    description="Endpoint racine qui retourne un message de bienvenue.",
    responses={200: {"description": "Succès"}, 500: {"description": "Erreur serveur"}},
)
async def root() -> dict[str, str]:
    """A simple endpoint that returns a greeting message."""

    return {"message": "Hello, Ronan!"}


# Déclarer un endpoint avec un param dans le path
@app.get(
    "/items/{item_id}",
    summary="Obtenir un élément",
    description="Endpoint pour obtenir un élément par son ID.",
    responses={200: {"description": "Succès"}, 404: {"description": "Élément non trouvé"}},
)
async def read_item(item_id: int) -> dict[str, int]:
    """Retrieve an item by its ID."""

    item = {"item_id": item_id}
    return item


# Déclarer un endpoint avec un query param
@app.get(
    "/search/",
    summary="Rechercher des éléments",
    description="Endpoint pour rechercher des éléments avec pagination.",
    responses={200: {"description": "Succès"}, 400: {"description": "Requête invalide"}},
)
# Les params déclarés sont obligatoire, ajouter | None pour ajouter d'autre paramètres optionnels
async def read_items(skip: int = 0, limit: int | None = None) -> dict[str, int | None]:
    """Retrieve items with pagination using query parameters."""
    response = {}
    response["skip"] = skip
    if limit is not None:
        response["limit"] = limit
    return response


# Déclarer un endpoint avec un body payload
@app.post(
    "/items/",
    summary="Créer un élément",
    description="Endpoint pour créer un nouvel élément.",
    responses={201: {"description": "Élément créé"}, 400: {"description": "Requête invalide"}},
)
# Il faut ajouter le modèle Pydantic importé comme type du paramètre 'payload'
async def create_item(payload: ItemCreate) -> dict[str, str | None]:
    """Create a new item from the request body payload."""

    # Utiliser la méthode model_dump() pour convertir le modèle en dictionnaire
    return payload.model_dump()
