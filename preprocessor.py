import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 

def preprocessor(data):
        data = pd.read_csv(data)

        data = data[['overall', 'movement_reactions', 'mentality_composure', 'passing', 'potential', 'value_eur', 'release_clause_eur', 'dribbling', 'wage_eur', 'power_shot_power', 'physic', 'mentality_vision', 'attacking_short_passing']
]
        # Separate the features and target variable
        X = data.drop('overall', axis=1)   
        y = data['overall']

        # Initialize the iterative imputer
        imp = IterativeImputer()

        # Fit and transform the data
        X_imputed = imp.fit_transform(X)

        # Convert the imputed data back into a DataFrame
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

        return X_imputed_df, y