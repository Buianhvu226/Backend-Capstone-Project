import pandas as pd
from django.db.models import Q

def fetch_profiles_from_db():
    """Fetch all profiles from the Django ORM and return as a DataFrame."""
    try:
        # Import here to avoid circular imports
        from profiles.models import Profile
        
        queryset = Profile.objects.all().values(
            'id', 'title', 'full_name', 'born_year', 'losing_year', 'description',
            'name_of_father', 'name_of_mother', 'siblings', 'status', 'created_at', 'updated_at'
        )
        
        if not queryset.exists():
            print("Warning: No profiles found in database")
            return pd.DataFrame()
            
        df = pd.DataFrame(list(queryset))
        
        # Rename columns to match expected DataFrame structure
        df.rename(columns={
            'id': 'id',
            'title': 'Tiêu đề',
            'full_name': 'Họ và tên',
            'born_year': 'Năm sinh',
            'losing_year': 'Năm thất lạc',
            'description': 'Chi tiet_merged',  # Match DETAIL_COLUMN_NAME
            'name_of_father': 'Tên cha',
            'name_of_mother': 'Tên mẹ',
            'siblings': 'Anh chị em',
        }, inplace=True)
        
        # Reset index to ensure continuous indexing
        df.reset_index(drop=True, inplace=True)
        
        print(f"Successfully fetched {len(df)} profiles from database")
        return df
        
    except Exception as e:
        print(f"Error fetching profiles from database: {e}")
        return pd.DataFrame()

def find_similar_profiles(profile, top_k=5):
    """Find similar profiles to the given profile using vector search."""
    from .embedding import get_embedding, initialize_vector_db
    from .config import DETAIL_COLUMN_NAME
    from profiles.models import Profile as ProfileModel
    
    # Get the profile description
    description = getattr(profile, DETAIL_COLUMN_NAME, None)
    if not description:
        description = profile.description
        
    # Get embedding for the profile
    embedding = get_embedding(description, task_type="RETRIEVAL_DOCUMENT")
    if not embedding:
        print(f"Could not create embedding for profile {profile.id}")
        return []
        
    # Initialize vector DB
    collection = initialize_vector_db()
    if not collection:
        return []
        
    # Search for similar profiles
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k + 1,  # +1 because the profile itself might be in the results
            include=["metadatas", "distances", "documents"]
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
            
        # Get the IDs of similar profiles
        similar_ids = results['ids'][0]
        
        # Filter out the profile itself
        similar_ids = [int(id) for id in similar_ids if str(id) != str(profile.id)]
        
        # Limit to top_k
        similar_ids = similar_ids[:top_k]
        
        # Get the actual profile objects
        similar_profiles = list(ProfileModel.objects.filter(id__in=similar_ids))
        
        # Sort them in the same order as the similar_ids
        similar_profiles.sort(key=lambda p: similar_ids.index(p.id))
        
        return similar_profiles
        
    except Exception as e:
        print(f"Error searching for similar profiles: {e}")
        return []