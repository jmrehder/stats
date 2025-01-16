import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistik-Pakete
import scipy.stats as stats
import statsmodels.api as sm

# Für den PDF-Bericht
from fpdf import FPDF
from io import BytesIO

#########################################
# Hauptfunktion & Sidebar (7 Optionen)  #
#########################################
def main():
    st.title("Statistische Analysen mit Streamlit")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Wählen Sie eine Ansicht",
        [
            "Startseite",
            "Seaborn-Datensatz laden",
            "Datei-Upload",
            "Daten-Exploration",
            "Datenbereinigung",
            "Deskriptive Statistik",
            "Hypothesenmanager",
            "Statistische Analysen (t-Test, Korrelation, Chi², Regression)",
            "Vorhersagen",
            "PDF Bericht"
        ]
    )
    
    if app_mode == "Startseite":
        startseite()
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
    elif app_mode == "Hypothesenmanager":
        hypothesen_manager()
    elif app_mode == "Statistische Analysen (t-Test, Korrelation, Chi², Regression)":
        advanced_analyses()
    elif app_mode == "Vorhersagen":
        vorhersagen()
    elif app_mode == "PDF Bericht":
        pdf_bericht()


#########################################
# Funktion: Startseite                  #
#########################################
def startseite():
    st.header("Willkommen zu Statistische Analysen mit Streamlit")
    st.write(
        """
        Diese Anwendung bietet Ihnen eine interaktive Möglichkeit, statistische Analysen durchzuführen. 
        Wählen Sie in der linken Seitenleiste eine Funktion aus, um zu beginnen.

        **Funktionen:**
        - **Seaborn-Datensatz laden:** Nutzen Sie vordefinierte Datensätze von Seaborn für Ihre Analysen.
        - **Datei-Upload:** Laden Sie eigene CSV- oder Excel-Dateien hoch.
        - **Daten-Exploration:** Erkunden Sie Ihre Daten durch visuelle und statistische Zusammenfassungen.
        - **Datenbereinigung:** Entfernen Sie fehlende Werte und bereinigen Sie Ihren Datensatz.
        - **Deskriptive Statistik:** Erhalten Sie grundlegende Kennzahlen und Visualisierungen Ihrer Daten.
        - **Hypothesenmanager:** Erstellen Sie Null- und Alternativhypothesen basierend auf Ihren Variablen.
        - **Statistische Analysen:** Führen Sie t-Tests, Korrelationen, Chi²-Tests und Regressionen durch.
        - **Vorhersagen:** Nutzen Sie trainierte Regressionsmodelle, um Werte vorherzusagen.
        - **PDF Bericht:** Exportieren Sie Ihre Ergebnisse in einem strukturierten PDF-Bericht.

        **So funktioniert es:**
        1. Laden Sie Ihre Daten hoch oder wählen Sie einen vordefinierten Datensatz.
        2. Wählen Sie eine Funktion aus der Seitenleiste aus.
        3. Folgen Sie den Anweisungen auf dem Bildschirm, um Ihre Analysen durchzuführen.

        Viel Spaß bei der Nutzung dieser Anwendung!
        """
    )

#########################################
# Funktion: Seaborn-Datensätze laden    #
#########################################
def seaborn_datasets():
    st.header("Seaborn-Datensätze laden")
    st.info(
        """
        Hier können Sie einen vordefinierten Datensatz aus der Seaborn-Bibliothek laden.
        Wählen Sie einen Datensatz aus, um eine Beschreibung und den Datensatz anzuzeigen.
        """
    )
    # Dataset-Beschreibungen
    dataset_descriptions = {
        "anagrams": "Ergebnisse eines Experiments zur Untersuchung der Reaktionszeiten bei der Lösung von Anagrammen. 56 Zeilen, 3 Spalten",
        "anscombe": "Vier Datensätze, die zeigen, wie identische Statistiken unterschiedliche Verteilungen verdecken können. 44 Zeilen, 3 Spalten",
        "attention": "Ein Datensatz, der die Aufmerksamkeit von Probanden bei verschiedenen Aufgaben untersucht. 280 Zeilen, 3 Spalten",
        "brain_networks": "Informationen über die Netzwerke von Gehirnaktivitäten, die bei verschiedenen Hirnregionen gemessen wurden . 923 Zeilen, 63 Spalten",
        "car_crashes": "Daten über Verkehrsunfälle in verschiedenen US-Bundesstaaten. 51 Zeilen, 8 Spalten",
        "diamonds": "Ein Datensatz mit Preisen und Merkmalen von Diamanten. 53940 Zeilen, 10 Spalten",
        "dots": "Bewegungsdaten von Punkten auf einem Bildschirm. 180 Zeilen, 4 Spalten",
        "dowjones": "historische Aktienkurse, die zur Analyse der Performance des Aktienindex verwendet werden können. 649 Zeilen,2Spalten",
        "exercise": "Daten zu körperlichen Übungen und deren Auswirkungen auf die Gesundheit. 90 Zeilen, 4 Spalten",
        "flights": "Monatliche Fluggastzahlen über mehrere Jahre. 144 Zeilen, 3 Spalten",
        "fmri": "fMRT-Daten von Probanden unter verschiedenen Bedingungen. 1064 Zeilen, 5 Spalten",
        "geyser": "Die Wartezeit und Dauer zwischen Eruptionen des Geysirs Old Faithful im Yellowstone-Nationalpark. 272 Zeilen, 3 Spalten",
        "glue":" Die Fähigkeit von Modellen zur Verarbeitung und Analyse natürlicher Sprache, 64 Zeilen, 5 Spalten",
        "healthexp": "Datensatz untersucht den Zusammenhang zwischen Gesundheitsausgaben pro Kopf und der Lebenserwartung in verschiedenen Ländern. 274 Zeilen, 4 Spalten",
        "iris": "Messungen von Irisblumen (Länge und Breite von Kelch- und Blütenblättern). 150 Zeilen, 5 Spalten",
        "mpg": "Daten zur Kraftstoffeffizienz (miles per gallon) von Autos, einschließlich Eigenschaften wie Zylinderanzahl, Hubraum, Gewicht und Herkunft. 398 Zeilen, 9 Spalten",
        "penguins": "Daten zu verschiedenen Pinguinarten, einschließlich Gewicht und Flossenlänge. 344 Zeilen, 7 Spalten",
        "planets": "Entdeckte Exoplaneten mit ihren Eigenschaften. 1035 Zeilen, 6 Spalten",
        "seaice":"Untersucht die Veränderung der Meereisausdehnung über die Zeit, typischerweise in der Arktis oder Antarktis. 13175 Zeilen, 2 Spalten ",
        "taxis":"Beschreibt Fahrten mit Taxis, einschließlich Informationen wie Fahrtdauer, Distanz, Kosten, Trinkgeld und Zahlungsarten. 6433 Zeilen, 14",
        "tips": "Trinkgeld-Daten aus einem Restaurant. 244 Zeilen, 7 Spalten",
        "titanic": "Daten zu Passagieren der Titanic, einschließlich Überlebensstatus und Klassen. 891 Zeilen, 15 Spalten",
    }
    # Abrufen der Datensätze
    try:
        dataset_names = sns.get_dataset_names()
    except Exception as e:
        st.error("Fehler beim Abrufen der Datensätze. Bitte überprüfen Sie Ihre Seaborn-Installation.")
        return
    # Dropdown-Menü für die Auswahl eines Datensatzes
    selected_dataset = st.selectbox("Wählen Sie einen Datensatz", dataset_names)
    # Automatische Anzeige der Beschreibung
    description = dataset_descriptions.get(selected_dataset, "Keine Beschreibung verfügbar.")
    st.write(f"**Beschreibung von {selected_dataset}:**")
    st.write(description)
    # Datensatz laden und anzeigen
    if st.button("Datensatz laden"):
        try:
            df = sns.load_dataset(selected_dataset)
            st.session_state["df"] = df
            st.success(f"Datensatz '{selected_dataset}' erfolgreich geladen!")
            # Automatische Anzeige der Vorschau des Datensatzes
            st.write("Vorschau des Datensatzes:")
            preview_rows = st.slider("Anzahl der Vorschauzeilen", min_value=5, max_value=min(len(df), 100), value=5)
            st.dataframe(df.head(preview_rows))
        except Exception as e:
            st.error(f"Fehler beim Laden des Datensatzes '{selected_dataset}': {e}")

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
@st.cache_data
def load_data(file, file_type):
    """Lädt Daten aus einer CSV- oder Excel-Datei in einen DataFrame."""
    if file_type == "csv":
        return pd.read_csv(file)
    elif file_type == "excel":
        return pd.read_excel(file, sheet_name=None)  # Unterstützt mehrere Blätter


def file_uploader():
    st.header("Laden Sie einen Datensatz hoch")
    uploaded_file = st.file_uploader("Wählen Sie eine CSV- oder Excel-Datei aus", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            file_type = "csv"
        else:
            file_type = "excel"
        df = load_data(uploaded_file, file_type)
        if isinstance(df, dict):
            sheet_names = list(df.keys())
            selected_sheet = st.selectbox("Wählen Sie ein Arbeitsblatt", sheet_names)
            df = df[selected_sheet]
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
        st.warning("Bitte laden Sie zuerst einen Datensatz hoch.")
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
        st.warning("Bitte laden Sie zuerst einen Datensatz hoch.")
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
        st.warning("Bitte laden Sie zuerst einen Datensatz hoch.")
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
        column_to_plot = st.selectbox("Wählen Sie eine Spalte für das Histogramm", numeric_columns, key="histogram")
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
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Korrelationsmatrix")
        st.pyplot(fig_corr)
    else:
        st.warning("Keine numerischen Spalten für Korrelationsmatrix gefunden.")

    st.subheader("Barplot")
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    if categorical_columns:
        col = st.selectbox("Wählen Sie eine kategoriale Spalte für den Barplot", categorical_columns, key="barplot")
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Barplot von {col}")
        st.pyplot(fig)
    else:
        st.warning("Keine kategorialen Spalten für Barplot gefunden.")

    st.subheader("Stacked Barplot")
    if len(categorical_columns) >= 2:
        col_x = st.selectbox("Wählen Sie die erste kategoriale Variable", categorical_columns, key="stacked_bar_x")
        col_y = st.selectbox("Wählen Sie die zweite kategoriale Variable", [col for col in categorical_columns if col != col_x], key="stacked_bar_y")
        contingency_table = pd.crosstab(df[col_x], df[col_y])
        fig, ax = plt.subplots()
        contingency_table.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Gestapelter Barplot von {col_x} und {col_y}")
        st.pyplot(fig)
    else:
        st.warning("Mindestens zwei kategoriale Spalten für gestapelten Barplot erforderlich.")

    st.subheader("Weitere Visualisierungen")
    additional_plot = st.selectbox("Wählen Sie eine zusätzliche Visualisierung", ["Boxplot", "Scatterplot", "Violinplot", "Pairplot"])
    if additional_plot == "Boxplot":
        if numeric_columns:
            col = st.selectbox("Wählen Sie eine numerische Spalte für den Boxplot", numeric_columns, key="boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Boxplot von {col}")
            st.pyplot(fig)
        else:
            st.warning("Keine numerischen Spalten für Boxplot gefunden.")
    elif additional_plot == "Scatterplot":
        if len(numeric_columns) >= 2:
            col_x = st.selectbox("Wählen Sie die X-Achse", numeric_columns, key="scatter_x")
            col_y_options = [col for col in numeric_columns if col != col_x]
            col_y = st.selectbox("Wählen Sie die Y-Achse", col_y_options, key="scatter_y")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
            ax.set_title(f"Scatterplot: {col_x} vs. {col_y}")
            st.pyplot(fig)
        else:
            st.warning("Mindestens zwei numerische Spalten sind für einen Scatterplot erforderlich.")
    elif additional_plot == "Violinplot":
        if numeric_columns:
            col = st.selectbox("Wählen Sie eine numerische Spalte für den Violinplot", numeric_columns, key="violinplot")
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
    st.header("Statistische Analysen: t-Test, Korrelation, Chi², Regression, ANOVA")
    if "df" not in st.session_state:
        st.warning("Bitte laden Sie zuerst einen Datensatz hoch.")
        return
    df = st.session_state["df"]

    analysis_type = st.radio(
        "Welche Analyse möchten Sie durchführen?",
        ("t-Test", "Korrelation", "Chi²-Test", "Regression", "ANOVA")
    )

    if "current_hypothesis" in st.session_state:
        st.subheader("Gespeicherte Hypothese")
        st.write(f"**H₀:** {st.session_state['current_hypothesis']['null_hypothesis']}")
        st.write(f"**H₁:** {st.session_state['current_hypothesis']['alt_hypothesis']}")
    else:
        st.warning("Keine gespeicherte Hypothese. Bitte definieren Sie eine Hypothese im Hypothesenmanager.")

    # t-Test
    if analysis_type == "t-Test":
        st.subheader("t-Test (unabhängige Stichproben)")
        if st.button("Erklärung zum t-Test"):
            st.info("""
            **t-Test für unabhängige Stichproben**

            Der t-Test wird verwendet, um festzustellen, ob sich die Mittelwerte zweier Gruppen signifikant unterscheiden.
            Weitere Informationen: https://www.methodenberatung.uzh.ch/de/datenanalyse_spss/unterschiede/zentral/ttestabh.html
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
            Weitere Informationen: https://www.methodenberatung.uzh.ch/de/datenanalyse_spss/zusammenhaenge/korrelation.html
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

        # Cramér's V Berechnung
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        cramers_v = np.sqrt(phi2 / min(r - 1, k - 1))
        st.write(f"Cramér's V (Effektstärke): {cramers_v:.4f}")

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

            Die lineare Regression modelliert den Zusammenhang zwischen einer Zielvariablen (Y) und einer oder mehreren unabhängigen Variablen (X).
            Weitere Informationen: https://www.methodenberatung.uzh.ch/de/datenanalyse_spss/zusammenhaenge/ereg.html
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

            st.subheader("Ergebnisse der Regression")
            st.write(f"**R² (Bestimmtheitsmaß):** {model.rsquared:.4f}")
            st.write(f"**Adj. R² (Angepasstes Bestimmtheitsmaß):** {model.rsquared_adj:.4f}")
            st.write(f"**F-Statistik:** {model.fvalue:.4f}")
            st.write(f"**p-Wert der F-Statistik:** {model.f_pvalue:.4e}")

            st.write("**Deutung des p-Werts der F-Statistik:**")
            if model.f_pvalue < 0.001:
                st.info("Der p-Wert der F-Statistik ist sehr klein (p < 0.001), was darauf hinweist, dass das Modell insgesamt hochsignifikant ist.")
            elif model.f_pvalue < 0.05:
                st.info("Der p-Wert der F-Statistik ist signifikant (p < 0.05), was darauf hinweist, dass das Modell insgesamt statistisch signifikant ist.")
            else:
                st.warning("Der p-Wert der F-Statistik ist nicht signifikant (p ≥ 0.05), was darauf hinweist, dass das Modell möglicherweise keine gute Erklärung für die Zielvariable liefert.")

            st.write("**Deutung:**")
            if model.rsquared > 0.7:
                st.info("Das Modell erklärt einen hohen Anteil der Varianz in der Zielvariable.")
            elif model.rsquared > 0.4:
                st.info("Das Modell erklärt einen moderaten Anteil der Varianz in der Zielvariable.")
            else:
                st.info("Das Modell erklärt nur einen geringen Anteil der Varianz in der Zielvariable.")

            st.subheader("Koeffizienten")
            coef_table = pd.DataFrame({
                "Variable": model.params.index,
                "Koeffizient": model.params.values,
                "p-Wert": model.pvalues.values,
                "95% CI (Untergrenze)": model.conf_int()[0],
                "95% CI (Obergrenze)": model.conf_int()[1]
            })
            st.dataframe(coef_table)

            st.write("**Interpretation der Koeffizienten:**")
            for i, row in coef_table.iterrows():
                if row["Variable"] == "const":
                    st.write(f"Der konstante Term (Intercept) beträgt {row['Koeffizient']:.4f}.")
                else:
                    st.write(f"Eine Einheitserhöhung in {row['Variable']} führt zu einer Änderung von {row['Koeffizient']:.4f} in der Zielvariable.")

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

            st.subheader("Residualanalyse")
            data["Vorhersage"] = model.predict(X_const)
            data["Residuals"] = y - data["Vorhersage"]
            fig_residuals, ax_residuals = plt.subplots()
            ax_residuals.scatter(data["Vorhersage"], data["Residuals"], alpha=0.5)
            ax_residuals.axhline(0, color="red", linestyle="--")
            ax_residuals.set_xlabel("Vorhersagewerte")
            ax_residuals.set_ylabel("Residuen")
            ax_residuals.set_title("Residualplot: Vorhersagen vs. Residuen")
            st.pyplot(fig_residuals)

    # ANOVA (Varianzanalyse)
    elif analysis_type == "ANOVA":
        st.subheader("Varianzanalyse (ANOVA)")
        if st.button("Erklärung zur ANOVA"):
            st.info("""
            **ANOVA (Analysis of Variance)**

            Die ANOVA wird verwendet, um festzustellen, ob es signifikante Unterschiede zwischen den Mittelwerten von mehr als zwei Gruppen gibt.
            Weitere Informationen: https://www.methodenberatung.uzh.ch/de/datenanalyse_spss/unterschiede/zentral/mvarianz.html
            """)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
        if not numeric_columns or not categorical_columns:
            st.warning("Ihr Datensatz benötigt mindestens eine numerische und eine kategoriale Spalte.")
            return
        group_col = st.selectbox("Wählen Sie die Gruppenspalte (kategorial)", categorical_columns)
        numeric_col = st.selectbox("Wählen Sie die zu analysierende numerische Variable", numeric_columns)
        groups = [df[df[group_col] == group][numeric_col].dropna() for group in df[group_col].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        st.write(f"F-Statistik: {f_stat:.4f}")
        st.write(f"p-Wert: {p_value:.4e}")
        if p_value < 0.05:
            st.success("Es gibt signifikante Unterschiede zwischen den Gruppen (p < 0.05).")
        else:
            st.info("Keine signifikanten Unterschiede zwischen den Gruppen (p ≥ 0.05).")


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

######################################################################################
def pdf_bericht():
    st.header("PDF Bericht")
    st.info(
        """
        Wählen Sie die Inhalte aus, die in den PDF-Bericht aufgenommen werden sollen:
        - Deskriptive Statistiken
        - Korrelationsmatrix
        - Diagramme (z. B. Histogramme, Boxplots)
        - Ergebnisse eines ausgewählten Tests
        """
    )

    if "df" not in st.session_state:
        st.warning("Bitte lade zuerst einen Datensatz hoch.")
        return

    df = st.session_state["df"]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    # Checkboxen für Inhalte
    include_description = st.checkbox("Deskriptive Statistiken", value=True)
    include_correlation = st.checkbox("Korrelationsmatrix", value=True)
    include_histograms = st.checkbox("Histogramme", value=False)
    include_test_results = st.checkbox("Testergebnisse", value=True)

    # PDF Initialisierung
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Statistischer Bericht", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Generiert am: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.ln(10)

    # Inhalte hinzufügen
    if include_description:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Deskriptive Statistiken", ln=True)
        pdf.set_font("Courier", size=10)
        pdf.multi_cell(0, 10, df.describe().to_string())
        pdf.ln(10)

    if include_correlation and numeric_columns:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Korrelationsmatrix", ln=True)
        pdf.set_font("Courier", size=10)
        pdf.multi_cell(0, 10, df.corr(numeric_only=True).to_string())
        pdf.ln(10)

    if include_histograms and numeric_columns:
        for col in numeric_columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Histogram von {col}")
            plt.tight_layout()
            fig.savefig(f"{col}_histogram.png")
            plt.close(fig)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Histogram von {col}", ln=True)
            pdf.image(f"{col}_histogram.png", x=10, y=40, w=180)

    if include_test_results and "last_test" in st.session_state:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Ergebnisse des ausgewählten Tests", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, st.session_state["last_test"])

    # PDF speichern
    pdf_file = "bericht.pdf"
    pdf.output(pdf_file)

    # Download-Button
    with open(pdf_file, "rb") as f:
        st.download_button(
            "Download PDF Bericht",
            data=f,
            file_name="Statistischer_Bericht.pdf",
            mime="application/pdf"
        )


#########################################
# Funktion: Hypothesenmanager           #
#########################################
import streamlit as st
import pandas as pd
import numpy as np

#########################################
# Funktion: Hypothesenmanager           #
#########################################
def hypothesen_manager():
    st.header("Hypothesenmanager")

    # Prüfen, ob ein Datensatz vorhanden ist
    if "df" not in st.session_state:
        st.warning("Bitte laden Sie zuerst einen Datensatz hoch, bevor Sie Hypothesen definieren.")
        return

    # Datensatz laden und Spalten identifizieren
    df = st.session_state["df"]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    if not numeric_columns and not categorical_columns:
        st.error("Der Datensatz enthält keine gültigen Variablen.")
        return

    # Schritt 1: Typ der Hypothese auswählen
    st.subheader("Hypothesentyp auswählen")
    hypothesis_type = st.radio(
        "Wähle die Art der Hypothese:",
        ["t-Test", "Korrelation", "Chi²-Test", "Regression"]
    )

    # Hypothesenoptionen basierend auf der Auswahl
    st.subheader("Hypothesenformulierung")
    if hypothesis_type == "t-Test":
        if numeric_columns and categorical_columns:
            selected_numeric = st.selectbox("Wählen Sie eine numerische Variable", numeric_columns, key="ttest_numeric")
            selected_category = st.selectbox("Wählen Sie eine kategoriale Variable", categorical_columns, key="ttest_category")
            groups = df[selected_category].unique()
            if len(groups) != 2:
                st.warning("Für einen t-Test müssen genau zwei Gruppen vorhanden sein.")
                return

            st.markdown(f"**Automatisch generierte Hypothesen:**")
            st.write(f"**H₀:** Der Mittelwert von '{selected_numeric}' ist in beiden Gruppen gleich.")
            st.write(f"**H₁:** Der Mittelwert von '{selected_numeric}' unterscheidet sich zwischen den Gruppen.")

            if st.button("Hypothese speichern"):
                st.session_state["current_hypothesis"] = {
                    "type": "t-Test",
                    "null_hypothesis": f"Der Mittelwert von '{selected_numeric}' ist in beiden Gruppen gleich.",
                    "alt_hypothesis": f"Der Mittelwert von '{selected_numeric}' unterscheidet sich zwischen den Gruppen."
                }
                st.success("Hypothese wurde erfolgreich gespeichert!")
                st.write(f"**Gespeicherte Hypothese:**\nH₀: {st.session_state['current_hypothesis']['null_hypothesis']}\nH₁: {st.session_state['current_hypothesis']['alt_hypothesis']}")

    elif hypothesis_type == "Korrelation":
        if len(numeric_columns) >= 2:
            col_x = st.selectbox("Wählen Sie die erste numerische Variable", numeric_columns, key="corr_x")
            col_y = st.selectbox("Wählen Sie die zweite numerische Variable", [col for col in numeric_columns if col != col_x], key="corr_y")
            st.markdown(f"**Automatisch generierte Hypothesen:**")
            st.write(f"**H₀:** Es besteht keine lineare Korrelation zwischen '{col_x}' und '{col_y}'.")
            st.write(f"**H₁:** Es besteht eine lineare Korrelation zwischen '{col_x}' und '{col_y}'.")

            if st.button("Hypothese speichern"):
                st.session_state["current_hypothesis"] = {
                    "type": "Korrelation",
                    "null_hypothesis": f"Es besteht keine lineare Korrelation zwischen '{col_x}' und '{col_y}'.",
                    "alt_hypothesis": f"Es besteht eine lineare Korrelation zwischen '{col_x}' und '{col_y}'."
                }
                st.success("Hypothese wurde erfolgreich gespeichert!")
                st.write(f"**Gespeicherte Hypothese:**\nH₀: {st.session_state['current_hypothesis']['null_hypothesis']}\nH₁: {st.session_state['current_hypothesis']['alt_hypothesis']}")

    elif hypothesis_type == "Chi²-Test":
        if len(categorical_columns) >= 2:
            col_x = st.selectbox("Wählen Sie die erste kategoriale Variable", categorical_columns, key="chi2_x")
            col_y = st.selectbox("Wählen Sie die zweite kategoriale Variable", [col for col in categorical_columns if col != col_x], key="chi2_y")
            st.markdown(f"**Automatisch generierte Hypothesen:**")
            st.write(f"**H₀:** Es besteht keine Abhängigkeit zwischen '{col_x}' und '{col_y}'.")
            st.write(f"**H₁:** Es besteht eine Abhängigkeit zwischen '{col_x}' und '{col_y}'.")

            if st.button("Hypothese speichern"):
                st.session_state["current_hypothesis"] = {
                    "type": "Chi²-Test",
                    "null_hypothesis": f"Es besteht keine Abhängigkeit zwischen '{col_x}' und '{col_y}'.",
                    "alt_hypothesis": f"Es besteht eine Abhängigkeit zwischen '{col_x}' und '{col_y}'."
                }
                st.success("Hypothese wurde erfolgreich gespeichert!")
                st.write(f"**Gespeicherte Hypothese:**\nH₀: {st.session_state['current_hypothesis']['null_hypothesis']}\nH₁: {st.session_state['current_hypothesis']['alt_hypothesis']}")

    elif hypothesis_type == "Regression":
        if len(numeric_columns) >= 2:
            target_col = st.selectbox("Wähle die Zielvariable (Y)", numeric_columns, key="regression_y")
            features_possible = [col for col in numeric_columns if col != target_col]
            selected_features = st.multiselect("Wählen Sie unabhängige Variablen (X)", features_possible, key="regression_x")

            if selected_features:
                st.markdown(f"**Automatisch generierte Hypothesen:**")
                st.write(f"**H₀:** Die unabhängigen Variablen {', '.join(selected_features)} haben keinen Einfluss auf '{target_col}'.")
                st.write(f"**H₁:** Mindestens eine der unabhängigen Variablen {', '.join(selected_features)} hat einen Einfluss auf '{target_col}'.")

                if st.button("Hypothese speichern"):
                    st.session_state["current_hypothesis"] = {
                        "type": "Regression",
                        "null_hypothesis": f"Die unabhängigen Variablen {', '.join(selected_features)} haben keinen Einfluss auf '{target_col}'.",
                        "alt_hypothesis": f"Mindestens eine der unabhängigen Variablen {', '.join(selected_features)} hat einen Einfluss auf '{target_col}'."
                    }
                    st.success("Hypothese wurde erfolgreich gespeichert!")
                    st.write(f"**Gespeicherte Hypothese:**\nH₀: {st.session_state['current_hypothesis']['null_hypothesis']}\nH₁: {st.session_state['current_hypothesis']['alt_hypothesis']}")
            else:
                st.warning("Bitte wählen Sie mindestens eine unabhängige Variable aus.")

    # Hypothese validieren
    st.markdown("---")
    st.subheader("Hypothese validieren")
    st.info("Klicken Sie auf den Button unten, um die Hypothese zu testen und die Ergebnisse anzuzeigen.")

    if st.button("Hypothese testen"):
        if "current_hypothesis" in st.session_state:
            st.write(f"**Aktuelle Hypothese:**\nH₀: {st.session_state['current_hypothesis']['null_hypothesis']}\nH₁: {st.session_state['current_hypothesis']['alt_hypothesis']}")
        else:
            st.warning("Keine Hypothese gespeichert.")


#########################################
# Hauptblock                            #
#########################################
if __name__ == "__main__":
    main()
