"""Knowledge-base document management: list / upload / delete the source
docs that back retrieval. Protected by the same bearer-token dependency as
/chat, so only authenticated CRM users can manage the KB.

Upload accepts either a multipart file (.md/.txt) or a JSON body
{"name", "content"} -- both are handled from the raw Request since FastAPI
can't declare a File param and a JSON Body on the same route.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.services import ingest
from app.services.auth import AuthedUser, get_current_user

router = APIRouter(prefix="/kb")


@router.get("/documents")
def list_documents(user: AuthedUser = Depends(get_current_user)):
    return ingest.list_docs()


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, user: AuthedUser = Depends(get_current_user)):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing 'file' field")
        raw_name = file.filename or ""
        content = (await file.read()).decode("utf-8", errors="replace")
    else:
        body = await request.json()
        raw_name = body.get("name", "")
        content = body.get("content", "")

    try:
        return ingest.upsert_doc(raw_name, content)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.delete("/documents/{name}")
def delete_document(name: str, user: AuthedUser = Depends(get_current_user)):
    try:
        return ingest.delete_doc(name)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except LookupError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document named {name!r}")
