from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .embedding import initialize_vector_db
from .db_utils import fetch_profiles_from_db
from .search import search_combined_chroma

class ProfileSearchAPIView(APIView):
    """
    API endpoint for searching profiles using a user query.
    POST body: { "query": "..." }
    """
    def post(self, request):
        user_query = request.data.get("query", "").strip()
        if not user_query:
            return Response({"error": "Missing or empty 'query'."}, status=status.HTTP_400_BAD_REQUEST)

        # Initialize vector DB and fetch profiles
        collection = initialize_vector_db()
        if collection is None:
            return Response({"error": "Vector DB unavailable."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        df = fetch_profiles_from_db()
        if df.empty:
            return Response({"error": "No profiles found in database."}, status=status.HTTP_404_NOT_FOUND)

        # Run the search
        try:
            # search_combined_chroma returns a list of IDs (as strings)
            id_list = search_combined_chroma(
                df, collection, user_query, top_n_final=22, return_json=True, user=self.request.user
            )
            if not id_list:
                return Response({"results": []})

            # For each ID, build a detailed result dict
            detailed_results = []
            for idx in id_list:
                # idx may be string or int, ensure correct type for DataFrame lookup
                try:
                    profile = df.loc[int(idx)] if int(idx) in df.index else df[df['id'] == int(idx)].iloc[0]
                except Exception:
                    continue

                detailed_results.append({
                    "id": profile.get('id', ''),
                    "title": profile.get('Tiêu đề', ''),
                    "full_name": profile.get('Họ và tên', ''),
                    "losing_year": profile.get('Năm thất lạc', ''),
                    "born_year": profile.get('Năm sinh', ''),
                    "name_of_father": profile.get('Tên cha', ''),
                    "name_of_mother": profile.get('Tên mẹ', ''),
                    "siblings": profile.get('Anh chị em', ''),
                    "detail": str(profile.get('Chi tiet_merged', '')),
                })

            print("Tìm theo truy vấn detailed_results:", detailed_results)

            return Response({"results": detailed_results})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)