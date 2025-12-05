import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurazione pagina
st.set_page_config(
    page_title="Video Game Sales Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caricamento dati con cache
@st.cache_data
def load_data():
    """Carica e prepara i dati"""
    df = pd.read_csv("data/vgsales_clean.csv")
    df_pulito = df[df['Year_of_Release'] <= 2015].copy()
    
    # Crea colonne target e feature
    df_pulito.loc[:, 'Hit'] = (df_pulito['Global_Sales'] >= 1).astype(int)
    
    # Feature engineering
    publisher_avg_sales = df_pulito.groupby('Publisher')['Global_Sales'].mean().to_dict()
    df_pulito.loc[:, 'Publisher_Avg_Sales'] = df_pulito['Publisher'].map(publisher_avg_sales)
    
    developer_stats = df_pulito.groupby('Developer').agg({'Hit': ['sum', 'count']}).reset_index()
    developer_stats.columns = ['Developer', 'Hit_Count', 'Total_Count']
    developer_stats['Developer_Hit_Rate'] = developer_stats['Hit_Count'] / developer_stats['Total_Count']
    developer_hit_rate_dict = dict(zip(developer_stats['Developer'], developer_stats['Developer_Hit_Rate']))
    df_pulito.loc[:, 'Developer_Hit_Rate'] = df_pulito['Developer'].map(developer_hit_rate_dict)
    
    platform_stats = df_pulito.groupby('Platform').agg({'Hit': ['sum', 'count']}).reset_index()
    platform_stats.columns = ['Platform', 'Hit_Count', 'Total_Count']
    platform_stats['Platform_Hit_Rate'] = platform_stats['Hit_Count'] / platform_stats['Total_Count']
    platform_hit_rate_dict = dict(zip(platform_stats['Platform'], platform_stats['Platform_Hit_Rate']))
    df_pulito.loc[:, 'Platform_Hit_Rate'] = df_pulito['Platform'].map(platform_hit_rate_dict)
    
    return df_pulito

@st.cache_resource
def train_model(df):
    """Addestra il modello di predizione con Rating"""
    features = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating',
                'Publisher_Avg_Sales', 'Developer_Hit_Rate', 'Platform_Hit_Rate']
    
    df_model = df[features + ['Hit']].dropna()
    
    # Encoding
    label_encoders = {}
    X = df_model[features].copy()
    
    for col in ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    y = df_model['Hit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    return model, label_encoders, accuracy, features

# Carica dati e modello
df = load_data()
model, encoders, accuracy, features = train_model(df)

# ============ SIDEBAR ============
st.sidebar.title("🎮 Video Game Analytics")
st.sidebar.markdown("### Analisi Mercato Videogiochi")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigazione",
    ["📈 Analisi Dati", "📊 Dashboard Executive", "Modello Predittivo", "🤖 Predittore Hit"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Dataset**: 16.000+ giochi (1980-2015)
    
    **HIT**: Vendite ≥ 1M copie
    
    **Modello**: Logistic Regression
    
    **Accuracy**: {:.1%}
    """.format(accuracy)
)

# ============ PAGINA 1: DASHBOARD EXECUTIVE ============
if page == "📊 Dashboard Executive":
    st.title("📊 Dashboard Executive: Video Game Market")
    st.markdown("### Panoramica rapida del mercato videogiochi (2000-2015)")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = df['Global_Sales'].sum()
    total_games = len(df)
    hit_rate = (df['Hit'].sum() / len(df)) * 100
    avg_sales = df['Global_Sales'].mean()
    
    with col1:
        st.metric("📦 Vendite Totali", f"{total_sales:,.0f}M", "copie vendute")
    with col2:
        st.metric("🎮 Giochi Analizzati", f"{total_games:,}", "titoli")
    with col3:
        st.metric("🔥 Tasso Hit", f"{hit_rate:.1f}%", "≥1M copie")
    with col4:
        st.metric("💰 Media Vendite", f"{avg_sales:.2f}M", "per gioco")
    
    st.markdown("---")
    
    # Due colonne per grafici principali
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 📈 Trend Vendite nel Tempo")
        yearly = df.groupby('Year_of_Release')['Global_Sales'].sum().reset_index()
        yearly = yearly[yearly['Year_of_Release'].notna()]
        
        fig = px.line(yearly, x='Year_of_Release', y='Global_Sales',
                      labels={'Year_of_Release': 'Anno', 'Global_Sales': 'Vendite (M)'},
                      template='plotly_white')
        fig.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("📝 **Insight**: Il picco di vendite è stato raggiunto nel 2008-2010, seguito da un declino graduale dovuto alla transizione verso il digital gaming.")
    
    with col_right:
        st.markdown("#### 🎯 Top 10 Piattaforme per Vendite")
        platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(platform_sales, orientation='h',
                     labels={'value': 'Vendite (M)', 'Platform': 'Piattaforma'},
                     template='plotly_white', color=platform_sales.values,
                     color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("📝 **Insight**: PS2 e X360 dominano il mercato fisico. Le console Sony e Microsoft rappresentano oltre il 60% delle vendite totali.")
    
    st.markdown("---")
    
    # Grafico generi
    st.markdown("#### 🎨 Distribuzione Vendite per Genere")
    genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    
    fig = px.pie(values=genre_sales.values, names=genre_sales.index,
                 template='plotly_white', hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("📝 **Insight**: Action (20%), Sports (14%) e Shooter (12%) dominano il mercato. I generi d'azione attirano il pubblico più ampio.")
    
    # Note sui dati
    with st.expander("ℹ️ Note sui Dati e Metodologia"):
        st.markdown("""
        **Periodo Analizzato**: 1980-2015 (esclude giochi recenti ancora in fase di vendita attiva)
        
        **Dati Mancanti**: 
        - Developer: ~40% missing (comune per giochi vecchi)
        - Critic Score: ~30% missing (non tutti i giochi ricevono recensioni)
        
        **Definizione HIT**: Gioco con vendite globali ≥ 1 milione di copie (top ~13% del mercato)
        
        **Limitazioni**:
        - Solo vendite fisiche (no digital download)
        - Dati aggregati per regione (NA, EU, JP, Others)
        - Inflazione non considerata
        """)

# ============ PAGINA 2: PREDITTORE HIT ============
elif page == "🤖 Predittore Hit":
    st.title("🤖 Predittore di Successo: Il Tuo Gioco Sarà un HIT?")
    
    st.markdown("""
    ### 🎯 Cosa Significa "HIT"?
    
    Un gioco è considerato **HIT** quando raggiunge **almeno 1 milione di copie vendute** a livello globale.
    
    **Perché è importante?**
    - Solo il **13%** dei giochi raggiunge questo traguardo
    - Rappresenta il punto di break-even per la maggior parte dei progetti AAA
    - Indica un forte product-market fit e brand awareness
    
    ---
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Inserisci le Caratteristiche del Gioco")
        
        # Platform - con opzione inserimento manuale
        platform_list = ['📝 Inserisci Nuovo...'] + sorted(df['Platform'].dropna().unique().tolist())
        platform_select = st.selectbox(
            "🎮 Piattaforma di Lancio",
            options=platform_list,
            help="Scegli dalla lista o inserisci una nuova piattaforma"
        )
        
        if platform_select == '📝 Inserisci Nuovo...':
            platform = st.text_input("✍️ Nome Piattaforma", placeholder="Es: PS5, Xbox Series X, Nintendo Switch")
        else:
            platform = platform_select
        
        # Genre - solo da lista esistente
        genre = st.selectbox(
            "🎨 Genere",
            options=sorted(df['Genre'].dropna().unique()),
            help="Il genere principale del gioco"
        )
        
        # Publisher - con opzione inserimento manuale
        publisher_list = ['📝 Inserisci Nuovo...'] + sorted(df['Publisher'].dropna().unique().tolist())
        publisher_select = st.selectbox(
            "🏢 Publisher",
            options=publisher_list,
            help="Scegli dalla lista o inserisci un nuovo publisher"
        )
        
        if publisher_select == '📝 Inserisci Nuovo...':
            publisher = st.text_input("✍️ Nome Publisher", placeholder="Es: Sony Interactive, Microsoft Studios")
        else:
            publisher = publisher_select
        
        # Developer - con opzione inserimento manuale
        developer_list = ['📝 Inserisci Nuovo...'] + sorted(df['Developer'].dropna().unique().tolist())
        developer_select = st.selectbox(
            "👨‍💻 Developer",
            options=developer_list,
            help="Scegli dalla lista o inserisci un nuovo developer"
        )
        
        if developer_select == '📝 Inserisci Nuovo...':
            developer = st.text_input("✍️ Nome Developer", placeholder="Es: Naughty Dog, FromSoftware")
        else:
            developer = developer_select
        
        # Rating - da lista esistente
        rating = st.selectbox(
            "🔞 Rating ESRB",
            options=sorted(df['Rating'].dropna().unique()),
            help="La classificazione ESRB del gioco"
        )
        
        predict_button = st.button("🔮 Predici Successo", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Risultato Predizione")
        
        if predict_button:
            # Validazione input
            if not platform or not publisher or not developer:
                st.error("⚠️ Compila tutti i campi obbligatori!")
            else:
                # Prepara input
                input_df = pd.DataFrame({
                    'Platform': [platform],
                    'Genre': [genre],
                    'Publisher': [publisher],
                    'Developer': [developer],
                    'Rating': [rating]
                })
                
                # Aggiungi feature storiche (usa medie se non esistono nel dataset)
                input_df['Publisher_Avg_Sales'] = df[df['Publisher'] == publisher]['Publisher_Avg_Sales'].iloc[0] if publisher in df['Publisher'].values else df['Publisher_Avg_Sales'].mean()
                input_df['Developer_Hit_Rate'] = df[df['Developer'] == developer]['Developer_Hit_Rate'].iloc[0] if developer in df['Developer'].values else df['Developer_Hit_Rate'].mean()
                input_df['Platform_Hit_Rate'] = df[df['Platform'] == platform]['Platform_Hit_Rate'].iloc[0] if platform in df['Platform'].values else df['Platform_Hit_Rate'].mean()
                
                # Encoding con gestione di valori nuovi
                input_encoded = input_df.copy()
                unknown_categories = []
                
                for col in ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']:
                    try:
                        input_encoded[col] = encoders[col].transform(input_df[col].astype(str))
                    except:
                        # Valore non visto nel training - usa la categoria più frequente
                        input_encoded[col] = 0
                        unknown_categories.append(col)
                
                # Mostra warning se ci sono categorie nuove
                if unknown_categories:
                    st.info(f"ℹ️ Valori nuovi rilevati per: {', '.join(unknown_categories)}. Usando valori di default per la predizione.")
                
                # Predizione
                proba = model.predict_proba(input_encoded[features])[0]
                prediction = model.predict(input_encoded[features])[0]
                
                # Visualizzazione risultato
                st.markdown("---")
                
                if prediction == 1:
                    st.success("### 🔥 ALTO POTENZIALE DI HIT!")
                    st.markdown(f"#### Probabilità di Successo: **{proba[1]:.1%}**")
                else:
                    st.warning("### ⚠️ BASSO POTENZIALE DI HIT")
                    st.markdown(f"#### Probabilità di Successo: **{proba[1]:.1%}**")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilità HIT (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ffcccc"},
                            {'range': [30, 60], 'color': "#fff4cc"},
                            {'range': [60, 100], 'color': "#ccffcc"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Fattori chiave
                st.markdown("#### 🔑 Fattori Chiave:")
                st.markdown(f"""
                - **Rating ESRB**: {rating}
                - **Publisher Track Record**: Media vendite {input_df['Publisher_Avg_Sales'].values[0]:.2f}M
                - **Developer Success Rate**: {input_df['Developer_Hit_Rate'].values[0]:.1%} hit rate storico
                - **Platform Performance**: {input_df['Platform_Hit_Rate'].values[0]:.1%} hit rate sulla piattaforma
                """)
    
    # Esempi reali
    st.markdown("---")
    st.markdown("### 💡 Esempi Reali dal Dataset")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.success("#### ✅ Esempio HIT Confermato")
        hit_example = df[df['Hit'] == 1].sample(1).iloc[0]
        st.markdown(f"""
        **{hit_example['Name']}**
        - Platform: {hit_example['Platform']}
        - Genre: {hit_example['Genre']}
        - Publisher: {hit_example['Publisher']}
        - Vendite: **{hit_example['Global_Sales']:.2f}M** copie
        """)
    
    with col_ex2:
        st.error("#### ❌ Esempio NON-HIT")
        non_hit_example = df[df['Hit'] == 0].sample(1).iloc[0]
        st.markdown(f"""
        **{non_hit_example['Name']}**
        - Platform: {non_hit_example['Platform']}
        - Genre: {non_hit_example['Genre']}
        - Publisher: {non_hit_example['Publisher']}
        - Vendite: **{non_hit_example['Global_Sales']:.2f}M** copie
        """)
# ============ PAGINA 4: MODELLO PREDITTIVO ============
elif page == "Modello predittivo":
    st.title("📈 Modello predittivo")
    
    st.markdown("### Visualizzazione del modello utilizzato")
    
    

# ============ PAGINA 4: ANALISI DATI ============
elif page == "📈 Analisi Dati":
    st.title("📈 Analisi Approfondita dei Dati")
    
    st.markdown("### Esplora le relazioni tra variabili e performance di vendita")
    
    # Analisi comparativa
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Publisher Analysis", "🎮 Platform vs Genre (Interattiva)", "📊 Trend Storici", "❌ Dati mancanti"])
    
    with tab1:
        st.markdown("#### Top 15 Publisher per Performance")
        
        pub_stats = df.groupby('Publisher').agg({
            'Global_Sales': ['sum', 'mean', 'count'],
            'Hit': 'sum'
        }).reset_index()
        pub_stats.columns = ['Publisher', 'Total_Sales', 'Avg_Sales', 'Num_Games', 'Num_Hits']
        pub_stats['Hit_Rate'] = (pub_stats['Num_Hits'] / pub_stats['Num_Games'] * 100).round(1)
        pub_stats = pub_stats.sort_values('Total_Sales', ascending=False).head(15)
        
        fig = px.scatter(pub_stats, x='Avg_Sales', y='Hit_Rate', 
                        size='Total_Sales', hover_data=['Publisher', 'Num_Games'],
                        labels={'Avg_Sales': 'Media Vendite per Gioco (M)', 
                               'Hit_Rate': 'Tasso di Hit (%)'},
                        template='plotly_white',
                        color='Hit_Rate',
                        color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("📝 **Come leggere**: Ogni bolla è un publisher. Più grande = più vendite totali. In alto a destra = publisher più efficaci.")
    
    with tab2:
        st.markdown("#### 🎮 Heatmap Interattiva: Piattaforma vs Genere per Periodo Storico")
        
        # Menu a tendina per periodo
        period = st.selectbox(
            "📅 Seleziona il Periodo Storico:",
            options=['Genesis Era (< 2000)', 'Early Era (2000-2006)', 'Peak Era (2007-2012)', 'Modern Era (2013-2015)'],
            index=1,
            help="Analizza come la distribuzione delle vendite per genere varia tra piattaforme in diversi periodi storici"
        )
        
        # Definisci periodi
        periods = {
            'Genesis Era (< 2000)': (1980, 1999),
            'Early Era (2000-2006)': (2000, 2006),
            'Peak Era (2007-2012)': (2007, 2012),
            'Modern Era (2013-2015)': (2013, 2015)
        }
        
        start_year, end_year = periods[period]
        df_period = df[(df['Year_of_Release'] >= start_year) & (df['Year_of_Release'] <= end_year)]
        
        if len(df_period) > 0:
            # Pivot table
            platform_genre = df_period.pivot_table(
                values='Global_Sales', 
                index='Platform', 
                columns='Genre', 
                aggfunc='sum', 
                fill_value=0
            )
            
            # Normalizza per riga (% di vendite per genere su ogni piattaforma)
            platform_genre_norm = platform_genre.div(platform_genre.sum(axis=1), axis=0) * 100
            
            # Filtra solo top 10 piattaforme per leggibilità
            top_platforms = platform_genre_norm.sum(axis=1).nlargest(10).index
            platform_genre_norm = platform_genre_norm.loc[top_platforms]
            
            # Crea heatmap Plotly
            fig = go.Figure(data=go.Heatmap(
                z=platform_genre_norm.values,
                x=platform_genre_norm.columns,
                y=platform_genre_norm.index,
                colorscale='YlOrRd',
                text=np.round(platform_genre_norm.values, 1),
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                colorbar=dict(title="% Vendite"),
                hovertemplate='<b>%{y}</b> - %{x}<br>Percentuale: %{text:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'{period} - Piattaforma vs Genere: Percentuale Vendite',
                xaxis_title='Genere',
                yaxis_title='Piattaforma',
                height=600,
                font=dict(size=11),
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiche periodo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎮 Giochi Analizzati", f"{len(df_period):,}")
            with col2:
                st.metric("💰 Vendite Totali", f"{df_period['Global_Sales'].sum():.1f}M")
            with col3:
                st.metric("📈 Media per Gioco", f"{df_period['Global_Sales'].mean():.2f}M")
            
            st.caption("📝 **Come leggere**: La heatmap mostra la distribuzione percentuale delle vendite per genere su ogni piattaforma. Colori più caldi = maggiore concentrazione di vendite in quel genere.")
        else:
            st.warning(f"⚠️ Nessun dato disponibile per il periodo {start_year}-{end_year}")
    
    with tab3:
        st.markdown("#### 📊 Evoluzione del Mercato nel Tempo")
        
        # Grafico vendite per anno e genere
        yearly_genre = df.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()
        yearly_genre = yearly_genre[yearly_genre['Year_of_Release'].notna()]
        
        # Top 5 generi per vendite totali
        top_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(5).index
        yearly_genre_top = yearly_genre[yearly_genre['Genre'].isin(top_genres)]
        
        fig = px.area(yearly_genre_top, x='Year_of_Release', y='Global_Sales', 
                     color='Genre',
                     labels={'Year_of_Release': 'Anno', 'Global_Sales': 'Vendite (M)', 'Genre': 'Genere'},
                     template='plotly_white',
                     title='Evoluzione Vendite per Genere (Top 5)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("📝 **Insight**: Mostra come i generi più popolari hanno dominato il mercato nel tempo. L'Action rimane costantemente forte, mentre lo Sport cresce nei picchi delle console mainstream.")

    with tab4:
        st.markdown("#### ❌ Dati Mancanti")

        # Analisi Dati Mancanti per Rating e Recensioni

        # Colonne di interesse per rating e recensioni
        rating_review_cols = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Rating']

        # Calcola i dati mancanti in numeri assoluti e percentuale
        missing_analysis = pd.DataFrame({
            'Colonna': rating_review_cols,
            'Mancanti': [df[col].isnull().sum() for col in rating_review_cols],
            'Totale': len(df),
        })
        missing_analysis['Percentuale'] = (missing_analysis['Mancanti'] / missing_analysis['Totale'] * 100).round(2)
        missing_analysis['Disponibili'] = missing_analysis['Totale'] - missing_analysis['Mancanti']
        missing_analysis['% Disponibili'] = (100 - missing_analysis['Percentuale']).round(2)

        print("📊 ANALISI DATI MANCANTI - Rating e Recensioni")
        print("=" * 80)
        print(missing_analysis.to_string(index=False))
        print("\n")

        # Crea visualizzazione grafica
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Grafico 1: Dati mancanti in percentuale
        colors_missing = ['#E63946' if x > 50 else '#F77F00' if x > 20 else '#06A77D' for x in missing_analysis['Percentuale']]
        ax1.barh(missing_analysis['Colonna'], missing_analysis['Percentuale'], color=colors_missing, alpha=0.8)
        ax1.set_xlabel('% Dati Mancanti', fontsize=11, fontweight='bold')
        ax1.set_title('Dati Mancanti per Colonna (Rating & Recensioni)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Aggiungi etichette con valori
        for i, (col, pct, missing) in enumerate(zip(missing_analysis['Colonna'], missing_analysis['Percentuale'], missing_analysis['Mancanti'])):
            ax1.text(pct + 1, i, f"{pct:.1f}% ({missing:,} NaN)", va='center', fontsize=10)

        # Grafico 2: Dati disponibili vs mancanti (stacked bar)
        x_pos = range(len(missing_analysis))
        ax2.bar(x_pos, missing_analysis['% Disponibili'], label='Disponibili', color='#06A77D', alpha=0.8)
        ax2.bar(x_pos, missing_analysis['Percentuale'], bottom=missing_analysis['% Disponibili'], label='Mancanti', color='#E63946', alpha=0.8)

        ax2.set_ylabel('Percentuale (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Disponibilità Dati: Disponibili vs Mancanti', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(missing_analysis['Colonna'], rotation=45, ha='right')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')

        # Aggiungi percentuali sulle barre
        for i, (avail, miss) in enumerate(zip(missing_analysis['% Disponibili'], missing_analysis['Percentuale'])):
            if avail > 5:
                ax2.text(i, avail/2, f"{avail:.0f}%", ha='center', va='center', fontweight='bold', color='white', fontsize=9)
            if miss > 5:
                ax2.text(i, avail + miss/2, f"{miss:.0f}%", ha='center', va='center', fontweight='bold', color='white', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

