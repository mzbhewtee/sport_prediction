from preprocessor import preprocessor
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import pickle



def main():
    option = ["Upload file", "Insert manually"]
    choice = st.sidebar.selectbox("Option", option)

    if choice == "Upload file":

        with open('sport_pred.sav', 'rb') as f:
            model = pickle.load(f)


        st.title('Player Rating Prediction')


        data_file = st.file_uploader("Upload your dataset", type=["csv", "excel"])
        if data_file is not None:
            st.write(type(data_file))
            X_imputed_df, y = preprocessor(data_file)

            X_imputed_df['overall'] = y
            X_imputed_df.drop('overall', axis=1, inplace=True)

            model.fit(X_imputed_df, y)

            predictions = model.predict(X_imputed_df)
            mse = mean_squared_error(y, predictions)

        prediction_got = ''

        if st.button('Predict'):
            prediction_got = mse
        st.success(prediction_got)

    else:
        st.title('Player Rating Prediction')

        with open('sport_pred.sav', 'rb') as f:
            model = pickle.load(f)

        movement_reactions = st.number_input('movement_reactions')
        mentality_composure = st.number_input('mentality_composure')
        passing = st.number_input('passing')
        potential = st.number_input('potential')
        value_eur = st.number_input('value_eur')
        release_clause_eur = st.number_input('release_clause_eur')
        dribbling = st.number_input('dribbling')
        wage_eur = st.number_input('wage_eur')
        power_shot_power = st.number_input('power_shot_power')
        physic = st.number_input('physic')
        mentality_vision = st.number_input('mentality_vision')
        attacking_short_passing = st.number_input('attacking_short_passing')
        overall = st.number_input('overall')

        X_imputed_df = pd.DataFrame([[movement_reactions, mentality_composure, passing, potential, value_eur, release_clause_eur, dribbling, wage_eur, power_shot_power, physic, mentality_vision, attacking_short_passing]], columns=['movement_reactions', 'mentality_composure', 'passing', 'potential', 'value_eur', 'release_clause_eur', 'dribbling', 'wage_eur', 'power_shot_power', 'physic', 'mentality_vision', 'attacking_short_passing'])
        y = pd.DataFrame([overall], columns=['overall'])

        if st.button('Predict'):
            model.fit(X_imputed_df, y)
            prediction = model.predict(X_imputed_df)
            mse = mean_squared_error(y, prediction)
            
            st.success(prediction)
            st.success(mse)


if __name__ == '__main__':
    main()
