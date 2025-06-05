from django.urls import path
from . import views

urlpatterns = [
    # List and create missing reports
    path('reports/', views.RecentlyMissingReportListCreateView.as_view(), name='missing-report-list-create'),
    
    # Public missing reports list
    path('reports/public/', views.RecentlyMissingReportPublicListView.as_view(), name='missing-report-public-list'),

    path('my-reports/', views.MyReportsListView.as_view(), name='my-reports'),
    
    # Individual missing report operations
    path('reports/<int:pk>/', views.RecentlyMissingReportDetailView.as_view(), name='missing-report-detail'),
    
    # Upload image
    path('reports/<int:report_id>/upload_image/', views.upload_missing_report_image, name='upload-missing-report-image'),
    
    # Missing person matches
    path('reports/<int:report_id>/matches/', views.get_missing_report_matches, name='missing-report-matches'),
    path('matches/<int:match_id>/update_status/', views.update_match_status, name='update-match-status'),
    path('reports/analyze-matches/', views.analyze_matches_with_llm, name='analyze-matches-llm'), 
    path('statistics/', views.statistics, name='missing-report-statistics'),
]