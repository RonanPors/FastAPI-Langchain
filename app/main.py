from fastapi import FastAPI
from schemas.enums import ModelName
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


# ============================================
#        ENDPOINTS AVEC PARAMÈTRES PATH
# ============================================


# Params standard sans type
@app.get(
    "/items/{item_id}",
    summary="Obtenir un élément",
    description="Endpoint pour obtenir un élément par son ID.",
    responses={200: {"description": "Succès"}, 404: {"description": "Élément non trouvé"}},
)
async def get_item(item_id) -> dict[str, str]:
    """Retrieve an item by its ID."""

    item = {"item_id": item_id}
    return item


# Params avec type
@app.get(
    "/items2/{item_id}",
    summary="Obtenir un élément avec type d'élément",
    description="Endpoint pour obtenir un élément par son ID.",
    responses={200: {"description": "Succès"}, 404: {"description": "Élément non trouvé"}},
)
async def get_item_with_type(item_id: int) -> dict[str, int]:
    """Retrieve an item by its ID."""

    item = {"item_id": item_id}
    return item


# Params avec valeurs prédéfinies
## Utiliser Enum pour définir des valeurs prédéfinies
@app.get(
    "/models/{model_name}",
    summary="Obtenir un modèle prédéfini",
    description="Endpoint pour obtenir un modèle par son nom.",
    responses={200: {"description": "Succès"}, 404: {"description": "Modèle non trouvé"}},
)
async def get_model(model_name: ModelName) -> dict[str, str]:
    """Retrieve a predefined model by its name."""

    if model_name == ModelName.model_a:
        model = {"model_name": model_name, "description": "Ceci est le modèle A"}
    elif model_name == ModelName.model_b:
        model = {"model_name": model_name, "description": "Ceci est le modèle B"}
    else:
        model = {"model_name": model_name, "description": "Ceci est le modèle C"}
    return model


# Params avec valeur path (url)
@app.get(
    "/files/{file_path:path}",
    summary="Obtenir un fichier par son chemin",
    description="Endpoint pour obtenir un fichier en spécifiant son chemin complet.",
    responses={200: {"description": "Succès"}, 404: {"description": "Fichier non trouvé"}},
)
async def read_file(file_path: str) -> dict[str, str]:
    return {"file_path": file_path}


# ============================================
#        ENDPOINTS AVEC PARAMÈTRES QUERY
# ============================================


# Déclarer un endpoint avec un query param
@app.get(
    "/search",
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


# ============================================
#        ENDPOINTS AVEC BODY PAYLOAD
# ============================================


# Déclarer un endpoint avec un body payload
@app.post(
    "/payload",
    summary="Créer un élément",
    description="Endpoint pour créer un nouvel élément.",
    responses={201: {"description": "Élément créé"}, 400: {"description": "Requête invalide"}},
)
# Il faut ajouter le modèle Pydantic importé comme type du paramètre 'payload'
async def create_item(payload: ItemCreate) -> dict[str, str | None]:
    """Create a new item from the request body payload."""

    # Utiliser la méthode model_dump() pour convertir le modèle en dictionnaire
    return payload.model_dump()


# Déclarer un endpoint avec un body payload et des path params
@app.post(
    "/payload/{item_id}",
    summary="Créer un élément avec ID",
    description="Endpoint pour créer un nouvel élément avec un ID spécifié.",
    responses={201: {"description": "Élément créé"}, 400: {"description": "Requête invalide"}},
)
async def create_item_with_id(item_id: int, payload: ItemCreate) -> dict[str, str | int | None]:
    """Create a new item with a specified ID from the request body payload."""

    # on peut aussi utiliser l'opérateur de merge :
    # response = payload.model_dump()
    # response |= {"item_id": item_id}
    # return response

    return {**payload.model_dump(), "item_id": item_id}


# Déclarer un endpoint avec un body, path et query params
@app.post(
    "/payload/{item_id}/query",
    summary="Créer un élément avec ID et détails",
    description="Endpoint pour créer un nouvel élément avec un ID spécifié et des détails supp.",
    responses={201: {"description": "Élément créé"}, 400: {"description": "Requête invalide"}},
)
async def create_item_with_id_and_query(
    item_id: int, payload: ItemCreate, q: str | None = None
) -> dict[str, str | int | None]:
    """Create a new item with a specified ID and optional query parameter from the request body payload."""

    response = {**payload.model_dump(), "item_id": item_id}
    if q:
        response["q"] = q
    return response
