from supabase import create_client, Client
from app.config import settings

# Créer une instance du client Supabase
supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_KEY
)

def get_supabase() -> Client:
    """Retourne l'instance du client Supabase"""
    return supabase