from transformer import StepTransformer

if __name__ == "__main__":
    transformer = StepTransformer(['age', 'gender', 'daily_gaming_hours', 'game_genre',
       'primary_game', 'gaming_platform', 'sleep_hours', 'sleep_quality',
       'sleep_disruption_frequency', 'face_to_face_social_hours_weekly', 'academic_work_performance'])
    transformer.transform("./data/processed/Gaming and Mental Health.csv")

