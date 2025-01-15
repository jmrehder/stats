import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistik-Pakete
import scipy.stats as stats
import statsmodels.api as sm

# Für den PDF-Bericht


#########################################
# Hauptfunktion & Sidebar (7 Optionen)  #
#########################################
def main():
    st.title("Statistische Analysen mit Streamlit")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Wähle eine Ansicht",
        [
            "Seaborn-Datensatz laden",
            "Datei-Upload",
            "Daten-Exploration",
            "Datenbereinigung",
            "Deskriptive Statistik",
            "Statistische Analysen (t-Test, Korrelation, Chi², Regression)",
            "Vorhersagen",
            "PDF Bericht"
        ]
    )
    
    if app_mode == "Seaborn-Datensatz laden":
        seaborn_datasets()
    elif app_mode == "Datei-Upload":
        file_uploader()
    elif app_mode == "Daten-Exploration":
        data_exploration()
    elif app_mode == "Datenbereinigung":
        datenbereinigung()
    elif app_mode == "Deskriptive Statistik":
        descriptive_statistics()
    elif app_mode == "Statistische Analysen (t-Test, Korrelation, Chi², Regression)":
        advanced_analyses()
    elif app_mode == "Vorhersagen":
        vorhersagen()
    elif app_mode == "PDF Bericht":
        pdf_bericht()

#########################################
# Funktion: Seaborn-Datensätze laden    #
#########################################
def seaborn_datasets():
    st.header("Seaborn-Datensätze laden")
    st.info(
        """
        Hier können Sie einen vordefinierten Datensatz aus der Seaborn-Bibliothek laden.
        Klicken Sie auf den Button, um eine Beschreibung zu erhalten und den Datensatz zu laden.
        """
    )
    
    dataset_descriptions = {
        "anscombe": "Vier Datensätze, die zeigen, wie identische Statistiken unterschiedliche Verteilungen verdecken können.",
        "attention": "Ein Datensatz, der die Aufmerksamkeit von Probanden bei verschiedenen Aufgaben untersucht.",
        "car_crashes": "Daten über Verkehrsunfälle in verschiedenen US-Bundesstaaten.",
        "diamonds": "Ein Datensatz mit Preisen und Merkmalen von Diamanten.",
        "dots": "Bewegungsdaten von Punkten auf einem Bildschirm.",
        "exercise": "Daten zu körperlichen Übungen und deren Auswirkungen auf die Gesundheit.",
        "flights": "Monatliche Fluggastzahlen über mehrere Jahre.",
        "fmri": "fMRT-Daten von Probanden unter verschiedenen Bedingungen.",
        "gammas": "Daten zu Gammastrahlen-Messungen.",
        "iris": "Messungen von Irisblumen (Länge und Breite von Kelch- und Blütenblättern).",
        "penguins": "Daten zu verschiedenen Pinguinarten, einschließlich Gewicht und Flossenlänge.",
        "planets": "Entdeckte Exoplaneten mit ihren Eigenschaften.",
        "tips": "Trinkgeld-Daten aus einem Restaurant.",
        "titanic": "Daten zu Passagieren der Titanic, einschließlich Überlebensstatus und Klassen.",
    }
    
    dataset_names = sns.get_dataset_names()
    
    selected_dataset = st.selectbox("Wähle einen Datensatz", dataset_names)
    if st.button("Beschreibung anzeigen"):
        description = dataset_descriptions.get(selected_dataset, "Keine Beschreibung verfügbar.")
        st.write(f"**Beschreibung von {selected_dataset}:**")
        st.write(description)
    
    if st.button("Datensatz laden"):
        df = sns.load_dataset(selected_dataset)
        st.session_state["df"] = df
        st.success(f"Datensatz '{selected_dataset}' erfolgreich geladen!")
        st.write("Vorschau des Datensatzes:")
        st.dataframe(df.head())

#########################################
# Funktion: Daten laden                 #
#########################################
@st.cache_data
def load_data(file, file_type):
    """Lädt Daten aus einer CSV- oder Excel-Datei in einen DataFrame."""
    if file_type == "csv":
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

#########################################
# Funktion: Datei-Upload                #
#########################################
def file_uploader():
    st.header("Lade einen Datensatz hoch")
    uploaded_file = st.file_uploader("Wähle eine CSV- oder Excel-Datei aus", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            file_type = "csv"
        else:
            file_type = "excel"
        df = load_data(uploaded_file, file_type)
        st.session_state["df"] = df
        st.success("Datei erfolgreich hochgeladen!")
        st.write("Vorschau des Datensatzes:")
        st.dataframe(df.head())

#########################################
# Funktion: Daten-Exploration           #
#########################################
def data_exploration():
    st.header("Erste Daten-Exploration")
    if "df" not in st.session_state:
        st.warning("Bitte lade zuerst einen Datensatz hoch.")
        return
    df = st.session_state["df"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Datensatz-Form")
        st.write("**Form:**", df.shape)
    with col2:
        st.markdown("### Spaltenliste")
        st.write(df.columns.tolist())
    
    st.markdown("---")
    st.subheader("Vorschau der Daten")
    num_rows = st.slider("Anzahl Zeilen zum Anzeigen", 1, 100, 5)
    st.dataframe(df.head(num_rows))
    
    st.markdown("---")
    st.subheader("Fehlende Werte")
    missing_values = df.isnull().sum()
    missing_count = missing_values[missing_values > 0]
    if missing_count.empty:
        st.success("Es wurden keine fehlenden Werte gefunden!")
    else:
        st.write("Fehlende Werte in den Spalten:")
        st.dataframe(missing_count.to_frame())

#########################################
# Funktion: Datenbereinigung            #
#########################################
def datenbereinigung():
    st.header("Datenbereinigung: Fehlende Werte entfernen")
    if "df" not in st.session_state:
        st.warning("Bitte lade zuerst einen Datensatz hoch.")
        return
    df = st.session_state["df"]
    st.write("Originale Anzahl der Zeilen:", df.shape[0])
    
    missing = df.isnull().sum()[df.isnull().sum() > 0]
    if missing.empty:
        st.success("Es wurden keine fehlenden Werte gefunden!")
        return
    st.write("Folgende Spalten enthalten fehlende Werte:")
    st.dataframe(missing.to_frame())
    columns_with_missing = list(missing.index)
    selected_cols = st.multiselect("Wählen Sie die Spalten, in denen Zeilen mit fehlenden Werten entfernt werden sollen:", columns_with_missing)
    if st.button("Leere Werte entfernen"):
        if not selected_cols:
            st.warning("Bitte wählen Sie mindestens eine Spalte aus.")
        else:
            df_clean = df.dropna(subset=selected_cols)
            st.write("Anzahl der Zeilen nach Entfernen:", df_clean.shape[0])
            st.dataframe(df_clean.head())
            st.success("Fehlende Werte wurden entfernt!")
            st.session_state["df"] = df_clean

#########################################
# Funktion: Deskriptive Statistik       #
#########################################
def descriptive_statistics():
    st.header("Deskriptive Statistik")
    if "df" not in st.session_state:
        st.warning("Bitte lade zuerst einen Datensatz hoch.")
        return
    df = st.session_state["df"]

    st.subheader("Basis-Statistiken")
    st.write(df.describe())
    st.info(
        """
        **Interpretation der Kennzahlen:**
        
        - **count**: Anzahl der nicht fehlenden Beobachtungen.
        - **mean**: Durchschnittswert der Variable.
        - **std**: Standardabweichung – misst, wie stark die Werte um den Mittelwert streuen.
        - **min** und **max**: Minimal- und Maximalwerte, die den Wertebereich darstellen.
        - **25%, 50%, 75%**: Quantile, wobei 50% dem Median entspricht.
        
        Diese Kennzahlen geben einen ersten Einblick in die Verteilung der numerischen Variablen.
        """
    )

    st.subheader("Histogramme")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        column_to_plot = st.selectbox("Wähle eine Spalte für das Histogramm", numeric_columns, key="histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[column_to_plot], kde=True, ax=ax)
        ax.set_title(f"Histogram von {column_to_plot}")
        st.pyplot(fig)
    else:
        st.warning("Keine numerischen Spalten für Histogramm gefunden.")

    st.subheader("Korrelationsmatrix")
    if numeric_columns:
        corr = df.corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Korrelationsmatrix")
        st.pyplot(fig_corr)
    else:
        st.warning("Keine numerischen Spalten für Korrelationsmatrix gefunden.")

    st.subheader("Weitere Visualisierungen")
    additional_plot = st.selectbox("Wähle eine zusätzliche Visualisierung", ["Boxplot", "Scatterplot", "Violinplot", "Pairplot"])
    if additional_plot == "Boxplot":
        if numeric_columns:
            col = st.selectbox("Wähle eine numerische Spalte für den Boxplot", numeric_columns, key="boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Boxplot von {col}")
            st.pyplot(fig)
        else:
            st.warning("Keine numerischen Spalten für Boxplot gefunden.")
    elif additional_plot == "Scatterplot":
        if len(numeric_columns) >= 2:
            col_x = st.selectbox("Wähle die X-Achse", numeric_columns, key="scatter_x")
            col_y_options = [col for col in numeric_columns if col != col_x]
            col_y = st.selectbox("Wähle die Y-Achse", col_y_options, key="scatter_y")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
            ax.set_title(f"Scatterplot: {col_x} vs. {col_y}")
            st.pyplot(fig)
        else:
            st.warning("Mindestens zwei numerische Spalten sind für einen Scatterplot erforderlich.")
    elif additional_plot == "Violinplot":
        if numeric_columns:
            col = st.selectbox("Wähle eine numerische Spalte für den Violinplot", numeric_columns, key="violinplot")
            fig, ax = plt.subplots()
            sns.violinplot(y=df[col], ax=ax)
            ax.set_title(f"Violinplot von {col}")
            st.pyplot(fig)
        else:
            st.warning("Keine numerischen Spalten für Violinplot gefunden.")
    elif additional_plot == "Pairplot":
        if len(numeric_columns) >= 2:
            st.write("Pairplot der numerischen Spalten:")
            pairplot_fig = sns.pairplot(df[numeric_columns].dropna())
            pairplot_fig.savefig("pairplot.png")
            st.image("pairplot.png")
        else:
            st.warning("Mindestens zwei numerische Spalten für Pairplot erforderlich.")

#########################################
# Funktion: Erweiterte Analysen         #
#########################################
def advanced_analyses():
    st.header("Statistische Analysen: t-Test, Korrelation, Chi², Regression")
    if "df" not in st.session_state:
        st.warning("Bitte lade zuerst einen Datensatz hoch.")
        return
    df = st.session_state["df"]

    analysis_type = st.radio(
        "Welche Analyse möchten Sie durchführen?",
        ("t-Test", "Korrelation", "Chi²-Test", "Regression")
    )

    # t-Test
    if analysis_type == "t-Test":
        st.subheader("t-Test (unabhängige Stichproben)")
        if st.button("Erklärung zum t-Test"):
            st.info("""
            **t-Test für unabhängige Stichproben**

            Der t-Test wird verwendet, um festzustellen, ob sich die Mittelwerte zweier Gruppen signifikant unterscheiden.
            **Anwendungsbeispiele:**
            - Vergleich des durchschnittlichen Gewichts von Männern und Frauen.
            - Vergleich von Testergebnissen zwischen zwei verschiedenen Schulklassen.

            **Voraussetzungen:**
            1. **Unabhängigkeit der Gruppen**: Die Messungen in einer Gruppe dürfen nicht von der anderen Gruppe beeinflusst werden.
            2. **Normalverteilung**: Die Zielvariable sollte in beiden Gruppen annähernd normalverteilt sein.
            3. **Varianzhomogenität**: Die Varianzen der beiden Gruppen sollten ähnlich sein. Falls nicht, kann ein angepasster t-Test (Welch-Test) verwendet werden.
            """)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
        if not numeric_columns or not categorical_columns:
            st.warning("Ihr Datensatz benötigt mindestens eine numerische und eine kategoriale Spalte.")
            return
        group_col = st.selectbox("Wählen Sie die Gruppenspalte (kategorial)", categorical_columns)
        numeric_col = st.selectbox("Wählen Sie die zu vergleichende numerische Variable", numeric_columns)
        groups = df[group_col].unique()
        if len(groups) != 2:
            st.warning("Für einen t-Test müssen genau zwei Gruppen vorhanden sein.")
            return
        group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
        group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
        t_stat, p_value = stats.ttest_ind(group1, group2)
        st.write(f"t-Statistik: {t_stat:.4f}")
        st.write(f"p-Wert: {p_value:.4e}")
        if p_value < 0.05:
            st.success("Die Differenz zwischen den Gruppen ist statistisch signifikant (p < 0.05).")
        else:
            st.info("Kein statistisch signifikanter Unterschied zwischen den Gruppen (p ≥ 0.05).")

    # Korrelationsanalyse
    elif analysis_type == "Korrelation":
        st.subheader("Korrelationsanalyse (Pearson)")
        if st.button("Erklärung zur Korrelationsanalyse"):
            st.info("""
            **Pearson-Korrelation**

            Die Pearson-Korrelation misst die Stärke und Richtung eines linearen Zusammenhangs zwischen zwei numerischen Variablen.

            **Anwendungsbeispiele:**
            - Zusammenhang zwischen Körpergröße und Gewicht.
            - Zusammenhang zwischen Lernzeit und Testergebnis.

            **Interpretation des Korrelationskoeffizienten (r):**
            - **r = 1**: Perfekte positive Korrelation (wenn eine Variable steigt, steigt die andere proportional).
            - **r = -1**: Perfekte negative Korrelation (wenn eine Variable steigt, sinkt die andere proportional).
            - **r = 0**: Kein linearer Zusammenhang zwischen den Variablen.

            **Voraussetzungen:**
            1. **Linearität**: Es sollte ein linearer Zusammenhang zwischen den Variablen bestehen.
            2. **Normalverteilung**: Beide Variablen sollten annähernd normalverteilt sein.
            """)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_columns) < 2:
            st.warning("Für eine Korrelationsanalyse benötigen Sie mindestens zwei numerische Spalten.")
            return
        col_x = st.selectbox("Wählen Sie die erste Variable (X)", numeric_columns)
        col_y = st.selectbox("Wählen Sie die zweite Variable (Y)", [col for col in numeric_columns if col != col_x])
        corr, p_value = stats.pearsonr(df[col_x], df[col_y])
        st.write(f"Korrelationskoeffizient (Pearson): {corr:.4f}")
        st.write(f"p-Wert: {p_value:.4e}")
        if p_value < 0.05:
            st.success("Die Korrelation ist statistisch signifikant (p < 0.05).")
        else:
            st.info("Die Korrelation ist nicht statistisch signifikant (p ≥ 0.05).")

    # Chi²-Test
    elif analysis_type == "Chi²-Test":
        st.subheader("Chi²-Test für Unabhängigkeit")
        if st.button("Erklärung zum Chi²-Test"):
            st.info("""
            **Chi²-Test für Unabhängigkeit**

            Der Chi²-Test prüft, ob ein Zusammenhang zwischen zwei kategorialen Variablen besteht.

            **Anwendungsbeispiele:**
            - Besteht ein Zusammenhang zwischen Geschlecht und Berufswahl?
            - Gibt es eine Abhängigkeit zwischen Rauchen (ja/nein) und Auftreten von Krankheiten (ja/nein)?

            **Voraussetzungen:**
            1. Die Daten müssen in einer Kreuztabelle zusammengefasst sein.
            2. Erwartete Häufigkeiten in jeder Zelle der Kreuztabelle sollten mindestens 5 betragen.

            **Interpretation:**
            - Ein niedriger p-Wert (p < 0.05) deutet darauf hin, dass ein Zusammenhang zwischen den Variablen besteht.
            - Ein hoher p-Wert (p ≥ 0.05) deutet darauf hin, dass die Variablen unabhängig sind.
            """)

        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
        if len(categorical_columns) < 2:
            st.warning("Für einen Chi²-Test benötigen Sie mindestens zwei kategoriale Spalten.")
            return
        col_x = st.selectbox("Wählen Sie die erste kategoriale Variable", categorical_columns)
        col_y = st.selectbox("Wählen Sie die zweite kategoriale Variable", [col for col in categorical_columns if col != col_x])
        contingency_table = pd.crosstab(df[col_x], df[col_y])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        st.write(f"Chi²-Wert: {chi2:.4f}")
        st.write(f"p-Wert: {p_value:.4e}")
        if p_value < 0.05:
            st.success("Die Variablen sind statistisch signifikant abhängig (p < 0.05).")
        else:
            st.info("Es gibt keine statistisch signifikante Abhängigkeit zwischen den Variablen (p ≥ 0.05).")

    # Regression
    elif analysis_type == "Regression":
        st.subheader("Lineare Regression (OLS)")
    if st.button("Erklärung zur Regression"):
        st.info("""
        **Lineare Regression**

        Die lineare Regression modelliert den Zusammenhang zwischen einer Zielvariablen (Y) und einer oder mehreren
        unabhängigen Variablen (X), um Vorhersagen zu treffen oder die Beziehung zu verstehen.

        **Anwendungsbeispiele:**
        - Vorhersage des Einkommens basierend auf Bildungsjahren.
        - Zusammenhang zwischen Werbebudget und Verkaufszahlen.

        **Voraussetzungen:**
        1. **Linearität**: Der Zusammenhang zwischen den unabhängigen Variablen und der Zielvariablen sollte linear sein.
        2. **Homoskedastizität**: Die Streuung der Residuen (Fehler) sollte über den Wertebereich konstant sein.
        3. **Normalverteilung der Residuen**: Die Fehler sollten annähernd normalverteilt sein.
        4. **Unabhängigkeit der Beobachtungen**: Es sollte keine Autokorrelation zwischen den Fehlern bestehen.
        """)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        st.warning("Ihr Datensatz benötigt mindestens eine numerische Spalte.")
        return
    target_col = st.selectbox("Wählen Sie die Zielvariable (Y)", numeric_columns)
    features_possible = [col for col in numeric_columns if col != target_col]
    selected_features = st.multiselect("Wählen Sie eine/n oder mehrere Features (X)", features_possible)
    if st.button("Trainiere Regressionsmodell"):
        if not selected_features:
            st.warning("Bitte wählen Sie mindestens eine Feature-Spalte aus.")
            return
        X = df[selected_features]
        y = df[target_col]
        data = pd.concat([X, y], axis=1).dropna()
        X = data[selected_features]
        y = data[target_col]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        st.write("**Regressionszusammenfassung**")
        st.text(model.summary())
        st.session_state["regression_model"] = model
        st.session_state["regression_features"] = selected_features
        st.session_state["regression_target"] = target_col
        st.session_state["regression_X_const"] = X_const
        
        # Regressionslinie plotten
        if len(selected_features) == 1:
            feature = selected_features[0]
            fig, ax = plt.subplots()
            ax.scatter(data[feature], y, alpha=0.5, label="Datenpunkte")
            x_range = np.linspace(data[feature].min(), data[feature].max(), 100)
            x_range_df = pd.DataFrame({feature: x_range})
            x_range_df_const = sm.add_constant(x_range_df)
            y_pred_line = model.predict(x_range_df_const)
            ax.plot(x_range, y_pred_line, color="red", label="Regressionslinie")
            ax.set_xlabel(feature)
            ax.set_ylabel(target_col)
            ax.legend()
            st.pyplot(fig)
            st.info("""
            **Interpretation der Regressionslinie:**
            Die rote Linie zeigt, wie die Zielvariable (Y) mit der unabhängigen Variablen (X) zusammenhängt.
            Ein steilerer Anstieg deutet auf eine stärkere Beziehung hin.
            """)

        # Homoskedastizität prüfen
        st.subheader("Homoskedastizität (Residualanalyse)")
        data["Vorhersage"] = model.predict(X_const)
        data["Residuals"] = y - data["Vorhersage"]
        fig_residuals, ax_residuals = plt.subplots()
        ax_residuals.scatter(data["Vorhersage"], data["Residuals"], alpha=0.5)
        ax_residuals.axhline(0, color="red", linestyle="--")
        ax_residuals.set_xlabel("Vorhersagewerte")
        ax_residuals.set_ylabel("Residuen")
        ax_residuals.set_title("Residualplot: Vorhersagen vs. Residuen")
        st.pyplot(fig_residuals)
        st.info("""
        **Interpretation des Residualplots:**
        - Eine zufällige Streuung der Residuen (ohne erkennbares Muster) deutet auf Homoskedastizität hin.
        - Falls ein Muster (z.B. Trichterform) erkennbar ist, könnte dies auf Heteroskedastizität hinweisen.
        Heteroskedastizität bedeutet, dass das Modell in bestimmten Bereichen weniger zuverlässig ist.
        """)


#########################################
# Funktion: Vorhersagen                 #
#########################################
def vorhersagen():
    st.header("Vorhersagen")
    st.info(
        """
        In diesem Bereich können Sie mit dem zuvor trainierten Regressionsmodell Vorhersagen für die Zielvariable treffen.
        """
    )
    if "regression_model" not in st.session_state:
        st.warning("Bitte trainieren Sie zuerst ein Regressionsmodell unter 'Statistische Analysen'.")
        return

    model = st.session_state["regression_model"]
    features = st.session_state["regression_features"]
    target_col = st.session_state["regression_target"]
    df = st.session_state["df"]
    
    input_data = {}
    for feature in features:
        col_min = float(df[feature].min())
        col_max = float(df[feature].max())
        default_val = float(df[feature].mean())
        input_data[feature] = st.number_input(
            f"Wert für {feature} (Bereich: {col_min} bis {col_max})",
            value=default_val,
            min_value=col_min,
            max_value=col_max
        )
    
    input_df = pd.DataFrame([input_data])
    
    # Sicherstellen, dass der konstante Term korrekt hinzugefügt wird
    input_df_const = sm.add_constant(input_df, has_constant='add')
    
    # Überprüfen, ob die Spalten übereinstimmen
    missing_cols = set(model.model.exog_names) - set(input_df_const.columns)
    for col in missing_cols:
        input_df_const[col] = 0  # Fehlende Spalten mit 0 auffüllen
    
    # Reihenfolge der Spalten anpassen
    input_df_const = input_df_const[model.model.exog_names]
    
    prediction = model.predict(input_df_const)
    st.write(f"Vorhergesagter Wert für {target_col}: {prediction[0]:.4f}")


#########################################
# Hauptblock                            #
#########################################
if __name__ == "__main__":
    main()
