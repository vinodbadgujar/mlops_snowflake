
import src.feature_engineering
import src.data_cleaning
import src.data_exploration 
import src.data_modeling


def main():
    #Data exploration
    print("Starting data exploration...")
    src.data_exploration.main()
    #Data cleaning
    print("Starting data cleaning...")
    src.data_cleaning.main()
    #Feature engineering
    print("Starting feature engineering...")    
    src.feature_engineering.main()
    #Data modeling
    print("Starting data modeling...")
    src.data_modeling.main()
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
