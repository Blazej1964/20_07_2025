import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
from PIL import Image
import base64
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
from datetime import datetime
import json
import re
import plotly.graph_objs as go
from hashlib import md5
from audiorecorder import audiorecorder  # type: ignore
from io import BytesIO
import numpy as np

# Ładowanie danych z pliku .env
env = dotenv_values(".env")

# Zmienne
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_COLLECTION_NAME = "notes"
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
FAVORITES_COLLECTION_NAME = "favorites" 
GALLERY_COLLECTION_NAME = "nazwa_twojej_kolekcji_w_galerii"

# Inicjalizacja klienta Qdrant
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"],
    api_key=env["QDRANT_API_KEY"],
)


qdrant_client = get_qdrant_client()

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

# Ochrona klucza API OpenAI
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI, aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Wprowadź klucz API:", type="password")

if not st.session_state.get("openai_api_key"):
    st.stop()

client = get_openai_client()

def assure_db_collection_exists():
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

def assure_gallery_collection_exists():
    if not qdrant_client.collection_exists(GALLERY_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=GALLERY_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"Kolekcja {GALLERY_COLLECTION_NAME} została utworzona.")

def generate_image_description(client, uploaded_file):
    uploaded_file.seek(0)
    try:
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        file_type = uploaded_file.type.split('/')[-1]
        image_url = f"data:image/{file_type};base64,{base64_image}"

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Wypisz tylko jakie składniki jedzenia widzisz na zdjęciu, nic więcej!"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Wystąpił błąd przy generowaniu opisu: {str(e)}"

def Kalorie(text):
    instructor_openai_client = get_openai_client()
    try:
        res = instructor_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Proszę obliczyć kalorie, białko i węglowodany dla poniższych składników:\n{text}. "
                               f"Wynik powinien być zwięzły i zawierać tylko wartości końcowe w formacie: "
                               f"Podsumowanie:\nKalorie = xxx kcal\nBiałko = xxx g\nWęglowodany = xxx g."
                },
            ],
        )

        response_text = res.choices[0].message.content.strip()

        # Debugowanie: Wyświetlenie odpowiedzi modelu
        st.write(f"Odpowiedź modelu: {response_text}")  # Zobacz, co zwraca model

        # Przetworzenie odpowiedzi na słownik
        if response_text:
            return parse_nutrition_response(response_text)
        else:
            return {"Kalorie": 0, "Białko": 0, "Węglowodany": 0}

    except Exception as e:
        return {"Kalorie": 0, "Białko": 0, "Węglowodany": 0}

def parse_nutrition_response(response):
    calories = re.search(r'Kalorie\s*=\s*(\d+)', response)
    protein = re.search(r'Białko\s*=\s*(\d+)', response)
    carbohydrates = re.search(r'Węglowodany\s*=\s*(\d+) ', response)

    return {
        "Kalorie": int(calories.group(1)) if calories else 0,
        "Białko": int(protein.group(1)) if protein else 0,
        "Węglowodany": int(carbohydrates.group(1)) if carbohydrates else 0
    }

def generate_embeddings(client, description):
    try:
        result = client.embeddings.create(
            input=[description],
            model=EMBEDDING_MODEL,
        )
        embedding = result.data[0].embedding
        return embedding
    except Exception as e:
        return f"Wystąpił błąd przy generowaniu embeddingu: {str(e)}"

def add_note_to_db(note_text, nutrition_info, uploaded_file, date_added, client, is_gallery=False):
    # Generuj wektory tylko w przypadku dodawania do ogólnej kolekcji
    vector = generate_embeddings(client, note_text) if not is_gallery else [0.0] * EMBEDDING_DIM

    if uploaded_file is not None:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        file_type = uploaded_file.type.split('/')[-1]
        image_url = f"data:image/{file_type};base64,{base64_image}"
    else:
        image_url = None  # Ustaw domyślną wartość lub inny odpowiedni sposób obsługi, gdy nie ma pliku

    # Użycie UUID jako unikalnego identyfikatora
    note_id = str(uuid.uuid4())

    try:
        qdrant_client.upsert(
            collection_name=GALLERY_COLLECTION_NAME if is_gallery else QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=note_id,
                    vector=vector,
                    payload={
                        "text": note_text,
                        "image": image_url,
                        "calories": nutrition_info["Kalorie"],
                        "protein": nutrition_info["Białko"],
                        "carbohydrates": nutrition_info["Węglowodany"],
                        "date_added": date_added.strftime("%Y-%m-%d")  # Zapisz datę w odpowiednim formacie
                    },
                )
            ]
        )
    except Exception as e:
        st.error(f"Wystąpił błąd podczas dodawania do galerii: {e}")

def list_notes_from_db(query=None, collection_name=QDRANT_COLLECTION_NAME):
    if not qdrant_client.collection_exists(collection_name):
        return []

    if not query:
        # Zmieniamy to na scrollowanie w zależności od podanej kolekcji
        notes = qdrant_client.scroll(collection_name=collection_name, limit=10)[0]
    else:
        query_vector = generate_embeddings(client, query)
        notes = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10,
        )

    return [
        {
            "id": note.id,
            "text": note.payload.get("text", ""),  # Użyj "" jako wartość domyślną, jeśli brak
            "image": note.payload.get("image"),     # Użyj get(), aby uniknąć błędów
            "calories": note.payload.get("calories", 0),  # Użyj 0 jako wartość domyślną
            "protein": note.payload.get("protein", 0),    # Użyj 0 jako wartość domyślną
            "carbohydrates": note.payload.get("carbohydrates", 0),  # Użyj 0 jako wartość domyślną
            "date_added": note.payload.get("date_added", "")  # Użyj "" jako wartość domyślną
        } for note in notes
    ]

def add_bmi_to_db(vector, payload, client):
    client.upsert(
        collection_name="bmi_data",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        ]
    )

def assure_db_collection_exists():
    if not qdrant_client.collection_exists("bmi_data"):
        qdrant_client.create_collection(
            collection_name="bmi_data",
            vectors_config=VectorParams(
                size=3072,
                distance=Distance.COSINE,
            ),
        )

def assure_favorites_collection_exists():
    if not qdrant_client.collection_exists(FAVORITES_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=FAVORITES_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"Kolekcja {FAVORITES_COLLECTION_NAME} została utworzona.")

def add_to_favorites(note_text, nutrition_info, uploaded_file, date_added, client):
    # Przygotowanie payloadu
    favorite_note = {
        "text": note_text,
        "calories": nutrition_info.get("Kalorie"),
        "protein": nutrition_info.get("Białko"),
        "carbohydrates": nutrition_info.get("Węglowodany"),
        "date_added": date_added.strftime("%Y-%m-%d")  # Formatuj datę
    }

    # Przeanalizowanie obrazu
    if uploaded_file is not None:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        file_type = uploaded_file.type.split('/')[-1]
        favorite_note["image"] = f"data:image/{file_type};base64,{base64_image}"
    else:
        favorite_note["image"] = None  # Ustaw domyślną wartość dla braku obrazu

    # Tworzymy wektor zerowy o długości 3072
    vector = [0.0] * EMBEDDING_DIM

    try:
        qdrant_client.upsert(
            collection_name=FAVORITES_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),  # Unikalny identyfikator
                    vector=vector,  # Dodaj wektor zerowy
                    payload=favorite_note
                )
            ]
        )
    except Exception as e:
        st.error(f"Wystąpił błąd podczas dodawania do ulubionych: {e}")
        print("Błąd w add_to_favorites:", e)

def delete_note_from_db(note_id, collection_name=QDRANT_COLLECTION_NAME):
    try:
        note_id_str = str(note_id)
        print(f"Próbuję usunąć notatkę o ID: {note_id_str} z kolekcji: {collection_name}")

        # Sprawdzenie, czy notatka istnieje
        existing_notes = list_notes_from_db(collection_name=collection_name)  # Użyj podanej kolekcji
        existing_note_ids = [str(note["id"]) for note in existing_notes]

        if note_id_str not in existing_note_ids:
            st.warning(f"Notatka o ID {note_id_str} nie istnieje w bazie danych.")
            return

        # Użycie poprawnej metody usuwania
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=[note_id_str]
        )
        st.success(f"Notatka o ID {note_id_str} została usunięta.")

        print(f"Notatka o ID {note_id_str} została pomyślnie usunięta.") 
    except Exception as e:
        print(f"Wystąpił błąd podczas usuwania notatki o ID {note_id}: {e}")
        st.error(f"Wystąpił błąd podczas usuwania notatki: {e}")

def calculate_caloric_needs(weight, height, age, sex='male'):
    if sex == 'male':
        BMR = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        BMR = 10 * weight + 6.25 * height - 5 * age - 161
    return BMR * 1.2  # Przyjmujemy, że jest to wartość dla siedzącego trybu życia

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )

    return transcript.text

def update_nutrition_in_notes(client):
    try:
        # Dla każdej notatki w bazie danych, zaktualizuj wartości pełne
        notes = list_notes_from_db()
        for note in notes:
            # Zaktualizuj payload notatki z nowymi wartościami z BMI
            qdrant_client.update(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=note["id"],
                        payload={
                            "calories": st.session_state.caloric_needs,
                            "protein": st.session_state.protein_needs,
                            "carbohydrates": st.session_state.carb_needs
                        }
                    )
                ]
            )
        st.success("Wartości odżywcze zostały zaktualizowane w notatkach.")
    except Exception as e:
        st.error(f"Wystąpił błąd podczas aktualizacji wartości odżywczych: {e}")

def get_nutrition_values():
    try:
        # Zakładam, że istnieje jedna notatka, z której chcesz wczytać wartości
        notes = list_notes_from_db()
        # Wczytaj pierwszą notatkę i jej wartości odżywcze (przykład)
        if notes:
            first_note = notes[0]
            return {
                'calories': first_note.get("calories", 0),
                'protein': first_note.get("protein", 0),
                'carbohydrates': first_note.get("carbohydrates", 0),
            }
        else:
            # Zwraca wartości domyślne, jeśli nie ma notatek
            return {'calories': 2000, 'protein': 150, 'carbohydrates': 300}
    except Exception as e:
        st.error(f"Wystąpił błąd podczas pobierania wartości odżywczych: {e}")
        return {'calories': 2000, 'protein': 150, 'carbohydrates': 300}  # Domyślne wartości
    
# Funkcja do zapisywania wartości do bazy danych
def save_nutrition_values(calories, protein, carbohydrates):
    try:
        # Wyszukaj notatki, aby zaktualizować
        notes = list_notes_from_db()
        if notes:
            # Załóżmy, że aktualizujemy tylko pierwszą notatkę
            note = notes[0]  # Możesz dostosować tę logikę według potrzeb

            # Przykładowa logika do generowania wektora na podstawie danych
            vector = generate_embeddings(client, f"Kalorie: {calories}, Białko: {protein}, Węglowodany: {carbohydrates}")

            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=note["id"],
                        vector=vector,  # Upewnij się, że wektor jest generowany
                        payload={
                            "calories": calories,
                            "protein": protein,
                            "carbohydrates": carbohydrates
                        }
                    )
                ]
            )
            st.success("Wartości odżywcze zostały zaktualizowane.")
        else:
            st.warning("Brak notatek do zaktualizowania.")
    except Exception as e:
        st.error(f"Wystąpił błąd podczas zapisywania wartości odżywczych: {e}")

# Główna część aplikacji
st.sidebar.markdown("# Wybierz opcję:")
selection = st.sidebar.selectbox("Wybierz opcję:", [
    "Wyniki dzienne",  # Galeria
    "Wpisz danie",  # Dodaj danie
    "Zdjęcie dania",  # Wczytaj danie
    "Ulubione",  # Ulubione
    "Znajdź danie",  # Wyszukaj notatkę
    
    #"Moje BMI",  # Oblicz BMI
    #"Wykresy"  # Nowa zakładka
])

# Resetowanie stanu sesji po zmianie zakładki
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

# Resetujemy sesję przy zmianie zakładki
if 'selected_option' not in st.session_state or st.session_state.selected_option != selection:
    st.session_state['uploaded_files'] = []
    st.session_state['selected_option'] = selection

if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0  # Inicjalizujemy klucz dla uploader'a

if selection == "Wykresy":
    st.header("Podsumowanie wyników dla wybranego dnia")

    # Wybór daty
    selected_date = st.date_input("Wybierz datę:", datetime.now())

    # Pobierz notatki dla wybranej daty
    notes = list_notes_from_db()  # Pobierz wszystkie notatki
    filtered_notes = [note for note in notes if note.get("date_added") == selected_date.strftime("%Y-%m-%d")]

    # Inicjalizacja zmiennych do sumowania składników
    total_calories = 0
    total_proteins = 0
    total_carbs = 0

    # Sumowanie wartości dla danego dnia
    for note in filtered_notes:
        total_calories += note.get('calories', 0)
        total_proteins += note.get('protein', 0)
        total_carbs += note.get('carbohydrates', 0)

    # Przygotowanie danych
    target_calories = st.session_state.get('caloric_needs', 0)
    target_proteins = st.session_state.get('protein_needs', 0)
    target_carbs = st.session_state.get('carb_needs', 0)

    # Wykresy
    # Wykres dla kalorii
    fig_calories = go.Figure()
    fig_calories.add_trace(go.Scatter(
        x=[selected_date.strftime("%Y-%m-%d")],
        y=[total_calories],
        mode='markers+lines',
        name='Kalorie',
        line=dict(width=2)
    ))
    fig_calories.add_hline(y=target_calories, line_color="red", line_dash="dash", annotation_text="Cel kalorii", annotation_position="top left")

    fig_calories.update_layout(
        title='Kalorie',
        xaxis_title='Data',
        yaxis_title='Kalorie',
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig_calories, use_container_width=True)

    # Wykres dla białka
    fig_proteins = go.Figure()
    fig_proteins.add_trace(go.Scatter(
        x=[selected_date.strftime("%Y-%m-%d")],
        y=[total_proteins],
        mode='markers+lines',
        name='Białko',
        line=dict(width=2)
    ))
    fig_proteins.add_hline(y=target_proteins, line_color="red", line_dash="dash", annotation_text="Cel białka", annotation_position="top left")

    fig_proteins.update_layout(
        title='Białko',
        xaxis_title='Data',
        yaxis_title='Białko (g)',
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig_proteins, use_container_width=True)

    # Wykres dla węglowodanów
    fig_carbs = go.Figure()
    fig_carbs.add_trace(go.Scatter(
        x=[selected_date.strftime("%Y-%m-%d")],
        y=[total_carbs],
        mode='markers+lines',
        name='Węglowodany',
        line=dict(width=2)
    ))
    fig_carbs.add_hline(y=target_carbs, line_color="red", line_dash="dash", annotation_text="Cel węglowodanów", annotation_position="top left")

    fig_carbs.update_layout(
        title='Węglowodany',
        xaxis_title='Data',
        yaxis_title='Węglowodany (g)',
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig_carbs, use_container_width=True)

# Kontynuacja poprzednich sekcji dla "Znajdź danie", "Wyniki dzienne", "Wpisz danie", "Moje BMI" i "Ulubione"

# Wyszukaj danie
elif selection == "Znajdź danie":
    query = st.text_input("Wyszukaj danie", on_change=None, key="search_query")

    if query and (st.button("Szukaj") or st.session_state.get("search_query") != ''):
        notes = list_notes_from_db(query)
        if notes:
            cols = st.columns(3)
            for i, note in enumerate(notes):
                with cols[i % 3]:
                    if note["image"]:
                        st.image(note["image"], caption="Miniaturka zdjęcia", use_container_width=True)
                    else:
                        st.write("Brak zdjęcia.")

        else:
            st.write("Brak pasujących notatek.")

if 'caloric_needs' not in st.session_state:
    st.session_state.caloric_needs = 0
if 'protein_needs' not in st.session_state:
    st.session_state.protein_needs = 0
if 'carb_needs' not in st.session_state:
    st.session_state.carb_needs = 0
if 'total_calories' not in st.session_state:
    st.session_state.total_calories = 0
if 'total_protein' not in st.session_state:
    st.session_state.total_protein = 0
if 'total_carbohydrates' not in st.session_state:
    st.session_state.total_carbohydrates = 0
if 'notes' not in st.session_state:
    st.session_state.notes = []

# Wpisz danie
# Wpisz danie
# Wpisz danie
if selection == "Wpisz danie":
    st.header("Dodaj notatkę audio:")

    # Inicjalizuj zmienne w session_state
    for key in ["note_audio_bytes_md5", "note_audio_bytes", "note_text", "note_audio_text", "transcription_done", "calories", "protein", "carbohydrates"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "transcription_done" else False

    # Przejrzysta strefa tekstowa, użyj wartości z session_state lub pustego ciągu
    st.session_state["note_text"] = st.text_area("Co dziś jadłeś:", value=st.session_state.get("note_text", ""), height=200)

    note_audio = audiorecorder(
        start_prompt="Nagraj notatkę",
        stop_prompt="Zatrzymaj nagrywanie",
    )

    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()

        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_bytes_md5"] = current_md5
            st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])
            st.session_state["note_text"] = st.session_state["note_audio_text"]
            st.session_state["transcription_done"] = True
            st.rerun()

    if st.button("Oblicz kalorie"):
        if st.session_state["note_text"]:
            nutrition_info = Kalorie(st.session_state["note_text"])
            st.session_state["calories"] = nutrition_info.get("Kalorie", "")
            st.session_state["protein"] = nutrition_info.get("Białko", "")
            st.session_state["carbohydrates"] = nutrition_info.get("Węglowodany", "")

            # Podsumowanie
            st.write("Podsumowanie:")
            st.write(f"Kalorie: {st.session_state['calories']} kcal")
            st.write(f"Białko: {st.session_state['protein']} g")
            st.write(f"Węglowodany: {st.session_state['carbohydrates']} g")
        else:
            st.warning("Proszę wprowadzić składniki w polu 'Co dziś jadłeś'.")

    # Zapisz notatki do bazy danych
    if st.session_state.get("calories") is not None:
        if st.button("Zapisz Danie"):
            assure_db_collection_exists()  # Upewnij się, że kolekcja istnieje

            now = datetime.now()
            note_text = st.session_state["note_text"]  # Tekst notatki

            nutrition_info = {
                "Kalorie": st.session_state["calories"],
                "Białko": st.session_state["protein"],
                "Węglowodany": st.session_state["carbohydrates"],
            }

            uploaded_file = None  # Jeśli nie ma pliku

            try:
                add_note_to_db(note_text, nutrition_info, uploaded_file, now, client)
                st.success("Danie zostało zapisane!")

                # Resetowanie stanu sesji po dodaniu dania
                # Ustawiamy wszystkie wartości na domyślne
                for key in ["note_audio_bytes_md5", "note_audio_bytes", "note_text", "note_audio_text", 
                             "transcription_done", "calories", "protein", "carbohydrates"]:
                    st.session_state[key] = None if key != "transcription_done" else False

                # Resetowanie wartości związanych z nagrywaniem
                st.session_state['uploaded_files'] = []  # Resetujemy wczytane pliki
                st.session_state['note_audio_text'] = ""  # Dodatkowe czyszczenie

                st.rerun()  # Odświeżenie strony

            except Exception as e:
                st.error(f"Wystąpił błąd podczas zapisywania notatki: {e}")



# Moje BMI
if selection == "Moje BMI":
    st.write("### Ustal BMI i zapotrzebowanie kaloryczne")

    # Sekcja ustawień BMI
    sex = st.selectbox("Płeć:", ['male', 'female'], index=0 if st.session_state.get('sex') != 'female' else 1)
    st.session_state.sex = sex  # Zapisz do session state

    # Predefiniowane wartości dla wagi i wzrostu
    weight_options = list(range(30, 201))  # Można dostosować zakres
    height_options = list(range(140, 221))  # Można dostosować zakres

    weight = st.selectbox("Waga (kg):", options=weight_options, index=weight_options.index(st.session_state.get('weight', 70)))
    st.session_state.weight = weight  # Zapisz do session state

    height = st.selectbox("Wzrost (cm):", options=height_options, index=height_options.index(st.session_state.get('height', 170)))
    st.session_state.height = height  # Zapisz do session state

    age = st.number_input("Wiek (lata):", min_value=0, value=st.session_state.get('age', 0))
    st.session_state.age = age  # Zapisz do session state

    goal_options = ["Normalne", "Redukcja wagi", "Nabranie masy mięśniowej"]
    goal = st.selectbox("Cel:", goal_options,
                         index=goal_options.index(st.session_state.get('goal', "Normalne")))
    st.session_state.goal = goal  # Zapisz do session state

    work_mode = st.selectbox("Wybierz tryb pracy:", 
                              ["Siedzący", "Mało aktywny", "Umiarkowanie aktywny", "Aktywny", "Bardzo aktywny"],
                              index=["Siedzący", "Mało aktywny", "Umiarkowanie aktywny", "Aktywny", "Bardzo aktywny"].index(st.session_state.get('work_mode', "Siedzący")))
    st.session_state.work_mode = work_mode  # Zapisz do session state

    activity_multiplier = {
        "Siedzący": 1.2,
        "Mało aktywny": 1.375,
        "Umiarkowanie aktywny": 1.55,
        "Aktywny": 1.725,
        "Bardzo aktywny": 1.9
    }

    if st.button("Oblicz zapotrzebowanie"):
        # Obliczania BMR na podstawie płci
        if sex == 'male':
            BMR = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            BMR = 10 * weight + 6.25 * height - 5 * age - 161

        # Obliczanie zapotrzebowania kalorycznego
        caloric_needs = BMR * activity_multiplier[work_mode]

        # Obliczanie zapotrzebowania na białko i węglowodany
        protein_needs = caloric_needs * 0.2 / 4  
        carb_needs = caloric_needs * 0.5 / 4    

        # Zaokrąglanie wartości do całkowitej
        caloric_needs = round(caloric_needs)
        protein_needs = round(protein_needs)
        carb_needs = round(carb_needs)

        # Informacja o obliczeniach
        st.success(f"Obliczone zapotrzebowanie: Kalorie = {caloric_needs}, Białko = {protein_needs}, Węglowodany = {carb_needs}")

        # Ustawienia wartości w session state
        st.session_state.caloric_needs = caloric_needs
        st.session_state.protein_needs = protein_needs
        st.session_state.carb_needs = carb_needs

    # Umożliwienie ręcznego wprowadzenia wartości
    caloric_needs = st.number_input("Zapotrzebowanie kaloryczne:", value=st.session_state.get('caloric_needs', 0), min_value=0)
    protein_needs = st.number_input("Zapotrzebowanie białka (g):", value=st.session_state.get('protein_needs', 0), min_value=0)
    carb_needs = st.number_input("Zapotrzebowanie węglowodanów (g):", value=st.session_state.get('carb_needs', 0), min_value=0)

    # Zapisz wartości do session state
    st.session_state.caloric_needs = caloric_needs
    st.session_state.protein_needs = protein_needs
    st.session_state.carb_needs = carb_needs

    if st.button("Zapisz dane"):
        try:
            # Sprawdzamy, czy istnieje już kolekcja
            assure_db_collection_exists()  # Funkcja do zapewnienia istnienia kolekcji

            # Przygotuj dane do zapisu
            bmi_data = {
                "weight": st.session_state.weight,
                "height": st.session_state.height,
                "age": st.session_state.age,
                "sex": st.session_state.sex,
                "goal": st.session_state.goal,
                "work_mode": st.session_state.work_mode,
                "caloric_needs": caloric_needs,
                "protein_needs": protein_needs,
                "carb_needs": carb_needs
            }

            # Generowanie wektora (przykład z zerami)
            vector = np.zeros(3072).tolist()

            # Dodanie punktu do bazy danych
            add_bmi_to_db(
                vector=vector,
                payload=bmi_data,
                client=qdrant_client
            ) 

            st.success("Dane zostały zapisane i wartości odżywcze zaktualizowane.", icon="👍")

        except Exception as e:
            st.error(f"Wystąpił błąd podczas zapisywania danych: {e}")

# Zdjęcie dania
if selection == "Zdjęcie dania":
    st.header("Wczytaj zdjęcia do galerii:")
    uploaded_files = st.file_uploader("Wybierz zdjęcia (maks. 5)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"uploader_{st.session_state['uploader_key']}")

    if uploaded_files:
        st.markdown("<h5>Wybierz datę zdjęcia:</h5>", unsafe_allow_html=True)
        selected_date = st.date_input("Data zdjęcia:", datetime.now(), key="selected_date")

        # Automatyczne generowanie opisów dla zdjęć
        for uploaded_file in uploaded_files:
            if 'generated_notes' not in st.session_state:
                st.session_state['generated_notes'] = {}

            if uploaded_file.name not in st.session_state['generated_notes']:
                description = generate_image_description(client, uploaded_file)
                if description and "Wystąpił błąd" not in description:
                    st.session_state['generated_notes'][uploaded_file.name] = description  

            # Wyświetlanie zdjęć, opisu oraz edytowalnego text_area
            image = Image.open(uploaded_file)
            st.image(image, caption='Wczytane zdjęcie', use_container_width=True)

            text_area_key = f"editable_note_{uploaded_file.name}"
            st.text_area(
                f"Edytuj notatkę dla {uploaded_file.name}:",
                value=st.session_state['generated_notes'].get(uploaded_file.name, ""),
                height=150,
                key=text_area_key 
            )

        # Obliczanie kalorii
        if st.button("Oblicz Kalorie"):
            all_ingredients = []
            for uploaded_file in uploaded_files:
                note_key = f"editable_note_{uploaded_file.name}"
                if note_key in st.session_state:
                    all_ingredients.append(st.session_state[note_key])

            combined_ingredients = ' '.join(all_ingredients).strip()

            if combined_ingredients:
                kalorie_result = Kalorie(combined_ingredients)
                st.session_state["kalorie_result"] = kalorie_result  
                st.success("Kalorie zostały obliczone.")

                if kalorie_result:
                    st.markdown(f"**Podsumowanie:**\n\n"
                                f"Kalorie = {kalorie_result['Kalorie']} kcal\n"
                                f"Białko = {kalorie_result['Białko']} g\n"
                                f"Węglowodany = {kalorie_result['Węglowodany']} g")
                else:
                    st.warning("Nie udało się obliczyć kalorii. Proszę spróbować ponownie.")

        # Zapisz notatki do bazy danych
        if st.session_state.get("kalorie_result"):
            col1, col2 = st.columns(2)  # Stwórz dwie kolumny dla przycisków

            with col1:
                if st.button("Zapisz Danie"):
                    assure_db_collection_exists()
                    for uploaded_file in uploaded_files:
                        edited_description = st.session_state.get(f"editable_note_{uploaded_file.name}", "")
                        nutrition_info = st.session_state.get("kalorie_result", {"Kalorie": 0, "Białko": 0, "Węglowodany": 0})

                        add_note_to_db(
                            note_text=edited_description,
                            nutrition_info=nutrition_info,
                            uploaded_file=uploaded_file,
                            date_added=selected_date,
                            client=client
                        )

                    st.success("Danie zostało zapisane.", icon="👍")  # Informacja o zapisie

                    # Resetowanie stanu sesji
                    st.session_state['generated_notes'] = {}
                    st.session_state["kalorie_result"] = None
                    st.session_state['uploaded_files'] = []  # Resetujemy wczytane pliki

                    # Zwiększamy klucz uploader'a, aby go zresetować
                    st.session_state['uploader_key'] += 1

                    # Przeskocz do zakładki "Wyniki dzienne"
                    st.session_state['selected_option'] = "Wyniki dzienne"
                    st.rerun()  # Odświeżenie strony

            with col2:
                if st.button("Zapisz Danie i dodaj do ulubionych"):
                    assure_db_collection_exists()  # Upewnij się, że kolekcja główna istnieje
                    assure_favorites_collection_exists()  # Upewnij się, że kolekcja ulubionych istnieje
                    for uploaded_file in uploaded_files:
                        edited_description = st.session_state.get(f"editable_note_{uploaded_file.name}", "")
                        nutrition_info = st.session_state.get("kalorie_result", {"Kalorie": 0, "Białko": 0, "Węglowodany": 0})

                        # Dodaj danie do galerii
                        add_note_to_db(
                            note_text=edited_description,
                            nutrition_info=nutrition_info,
                            uploaded_file=uploaded_file,
                            date_added=selected_date,
                            client=client
                        )

                        # Dodaj do ulubionych
                        add_to_favorites(
                            note_text=edited_description,
                            nutrition_info=nutrition_info,
                            uploaded_file=uploaded_file,
                            date_added=selected_date,
                            client=client
                        )

                    st.success("Danie zostało zapisane i dodane do ulubionych.", icon="👍")  # Informacja o zapisie

                    # Resetowanie stanu sesji
                    st.session_state['generated_notes'] = {}
                    st.session_state["kalorie_result"] = None
                    st.session_state['uploaded_files'] = []  # Resetujemy wczytane pliki

                    # Zwiększamy klucz uploader'a, aby go zresetować
                    st.session_state['uploader_key'] += 1

                    # Przeskocz do zakładki "Ulubione"
                    st.session_state['selected_option'] = "Ulubione"
                    st.rerun()  # Odświeżenie strony

# Ulubione
if selection == "Ulubione":
    st.header("Ulubione Dania")
    favorite_notes = list_notes_from_db(collection_name=FAVORITES_COLLECTION_NAME)

    if favorite_notes:
        cols = st.columns(3)  # Tworzymy 3 kolumny dla wyświetlania miniatur

        for i, note in enumerate(favorite_notes):
            with cols[i % 3]:  # Używanie operatora reszty do zmiany kolumny
                # Wyświetlanie miniatury zdjęcia
                st.image(note["image"], caption="Miniaturka zdjęcia", use_container_width=True)
                st.write(f"**Kalorie:** {note['calories']} kcal")
                st.write(f"**Białko:** {note.get('protein', 0)} g")
                st.write(f"**Węglowodany:** {note.get('carbohydrates', 0)} g")

                # Rozwijane okno dla składników
                with st.expander("Składniki", expanded=False):
                    st.text_area(
                        "Składniki:",
                        value=note['text'],
                        height=150,
                        disabled=True
                    )

                # Przycisk do usuwania potrawy z ulubionych
                if st.button("Usuń", key=f"remove_{note['id']}"):
                    try:
                        delete_note_from_db(note['id'], collection_name=FAVORITES_COLLECTION_NAME)
                        st.success("Potrawa została usunięta z ulubionych!")
                        # Nie odświeżaj strony, aby zachować widok na tej samej stronie
                    except Exception as e:
                        st.error(f"Wystąpił błąd: {e}")

                # Przycisk do dodania do galerii
                if st.button("Dodaj do galerii", key=f"add_{note['id']}"):
                    assure_gallery_collection_exists()  # Upewnij się, że kolekcja galerii istnieje
                    nutrition_info = {
                        "Kalorie": note['calories'],
                        "Białko": note['protein'],
                        "Węglowodany": note['carbohydrates'],
                    }

                    add_note_to_db(
                        note_text=note['text'],
                        nutrition_info=nutrition_info,
                        uploaded_file=None,  # brak przesyłanego pliku
                        date_added=datetime.now(),
                        client=client,
                        is_gallery=True  # Upewnij się, że ten argument jest przekazywany
                    )

                    st.success("Potrawa została dodana do galerii!", icon="👍")  # Informacja o dodaniu

                    # Resetowanie stanu sesji, jeżeli potrzebne
                    st.session_state['generated_notes'] = {}
                    st.session_state["kalorie_result"] = None

                    # Zmiana na zakładkę "Wyniki dzienne"
                    st.session_state['selected_option'] = "Wyniki dzienne"
                    st.rerun()  # Odświeżenie strony
    else:
        st.write("Brak ulubionych potraw.")

# Wyniki dzienne
if selection == "Wyniki dzienne":
    st.write("### Twoje dzienne zapotrzebowanie")

    # Wczytaj wartości odżywcze z bazy danych
    if 'nutrition_values' not in st.session_state:
        st.session_state.nutrition_values = get_nutrition_values()

    # Pola do edycji wartości odżywczych
    calories_input = st.number_input("Kalorie:", value=st.session_state.nutrition_values.get('calories', 2000), step=50)
    protein_input = st.number_input("Białko:", value=st.session_state.nutrition_values.get('protein', 150), step=5)
    carbohydrates_input = st.number_input("Węglowodany:", value=st.session_state.nutrition_values.get('carbohydrates', 300), step=5)

    if st.button("Zapisz wartości"):
        save_nutrition_values(calories_input, protein_input, carbohydrates_input)
        st.session_state.nutrition_values = {
            'calories': calories_input,
            'protein': protein_input,
            'carbohydrates': carbohydrates_input
        }

    # Pobierz notatki/miniaturki dla wybranej daty
    selected_date = st.date_input("Wybierz datę, aby wyświetlić zdjęcia:", datetime.now())
    notes = list_notes_from_db()  # Pobierz wszystkie notatki

    # Filtruj notatki według wybranej daty
    filtered_notes = [note for note in notes if note.get("date_added") == selected_date.strftime("%Y-%m-%d")]

    # Inicjalizacja zmiennych do sumowania składników
    st.session_state.total_calories = 0
    st.session_state.total_protein = 0
    st.session_state.total_carbohydrates = 0

    if filtered_notes:
        # Sortowanie notatek
        notes_with_images = [note for note in filtered_notes if note.get("image")]
        notes_without_images = [note for note in filtered_notes if not note.get("image")]
        sorted_notes = notes_with_images + notes_without_images

        for note in sorted_notes:
            st.session_state.total_calories += note.get('calories', 0)
            st.session_state.total_protein += note.get('protein', 0)
            st.session_state.total_carbohydrates += note.get('carbohydrates', 0)
            st.session_state.notes.append(note)  # Dodaj notatki do sesji

        # Obliczenia wartości do wykresów
        calories_consumed = min(st.session_state.total_calories, st.session_state.nutrition_values.get('calories', 0) or 0)
        calories_remaining = max(0, (st.session_state.nutrition_values.get('calories', 0) or 0) - st.session_state.total_calories)

        protein_consumed = min(st.session_state.total_protein, st.session_state.nutrition_values.get('protein', 0) or 0)
        protein_remaining = max(0, (st.session_state.nutrition_values.get('protein', 0) or 0) - st.session_state.total_protein)

        carbs_consumed = min(st.session_state.total_carbohydrates, st.session_state.nutrition_values.get('carbohydrates', 0) or 0)
        carbs_remaining = max(0, (st.session_state.nutrition_values.get('carbohydrates', 0) or 0) - st.session_state.total_carbohydrates)

        # Trójwymiarowy wykres kołowy dla kalorii
        st.write("<h3 style='text-align: center;'>Kalorie</h3>", unsafe_allow_html=True)
        fig_calories = go.Figure(data=[
            go.Pie(
                labels=['Zjadłeś', 'Pozostało'],
                values=[calories_consumed, calories_remaining],
                hole=.3,  # Użyj hole, aby uzyskać efekt donuta
                textinfo='label+percent',
                marker=dict(colors=['lightcoral', 'orange']),
                rotation=90,  # Rotacja dla trójwymiarowego efektu
                pull=[0.1, 0],  # Wysunięcie części wykresu
            )
        ])
        fig_calories.update_layout(title='Kalorie', showlegend=False)
        st.plotly_chart(fig_calories, use_container_width=True)

        # Wyśrodkowanie i pogrubienie informacji o spożyciu
        st.markdown(f"<div style='text-align: center;'>Zjadłeś już <strong style='font-size: 24px; color: red;'>{st.session_state.total_calories} kcal</strong> / <strong style='font-size: 24px; color: red;'>{st.session_state.nutrition_values.get('calories', 0) or 0} kcal</strong> dziennego zapotrzebowania</div>", unsafe_allow_html=True)

        # Oddzielenie wykresów linią
        st.markdown("<hr>", unsafe_allow_html=True)

        # Trójwymiarowy wykres kołowy dla białka
        st.write("<h3 style='text-align: center;'>Białko</h3>", unsafe_allow_html=True)
        fig_protein = go.Figure(data=[
            go.Pie(
                labels=['Zjadłeś', 'Pozostało'],
                values=[protein_consumed, protein_remaining],
                hole=.3,  # Użyj hole, aby uzyskać efekt donuta
                textinfo='label+percent',
                marker=dict(colors=['lightblue', 'blue']),
                rotation=90,  # Rotacja dla trójwymiarowego efektu
                pull=[0.1, 0],  # Wysunięcie części wykresu
            )
        ])
        fig_protein.update_layout(title='Białko', showlegend=False)
        st.plotly_chart(fig_protein, use_container_width=True)

        # Wyśrodkowanie i pogrubienie informacji o spożyciu
        st.markdown(f"<div style='text-align: center;'>Zjadłeś już <strong style='font-size: 24px; color: red;'>{st.session_state.total_protein} g</strong> / <strong style='font-size: 24px; color: red;'>{st.session_state.nutrition_values.get('protein', 0) or 0} g</strong> dziennego zapotrzebowania</div>", unsafe_allow_html=True)

        # Oddzielenie wykresów linią
        st.markdown("<hr>", unsafe_allow_html=True)

        # Trójwymiarowy wykres kołowy dla węglowodanów
        st.write("<h3 style='text-align: center;'>Węglowodany</h3>", unsafe_allow_html=True)
        fig_carbs = go.Figure(data=[
            go.Pie(
                labels=['Zjadłeś', 'Pozostało'],
                values=[carbs_consumed, carbs_remaining],
                hole=.3,  # Użyj hole, aby uzyskać efekt donuta
                textinfo='label+percent',
                marker=dict(colors=['lightgreen', 'green']),
                rotation=90,  # Rotacja dla trójwymiarowego efektu
                pull=[0.1, 0],  # Wysunięcie części wykresu
            )
        ])
        fig_carbs.update_layout(title='Węglowodany', showlegend=False)
        st.plotly_chart(fig_carbs, use_container_width=True)

        # Wyśrodkowanie i pogrubienie informacji o spożyciu z powiększoną czcionką i czerwonym kolorem
        st.markdown(f"<div style='text-align: center;'>Zjadłeś już <strong style='font-size: 24px; color: red;'>{st.session_state.total_carbohydrates} g</strong> / <strong style='font-size: 24px; color: red;'>{st.session_state.carb_needs} g</strong> dziennego zapotrzebowania</div>", unsafe_allow_html=True)

        # Oddzielenie wykresu od zdjęć linią
        st.markdown("<hr>", unsafe_allow_html=True)

        # Wyświetlanie zdjęć
        cols = st.columns(3)

        # Wyświetl zdjęcia
        for i, note in enumerate(notes_with_images):
            with cols[i % 3]:
                st.image(note["image"], width=150)
                st.write(f"**Kalorie:** {note['calories']} kcal")
                st.write(f"**Białko:** {note.get('protein', 0)} g")
                st.write(f"**Węglowodany:** {note.get('carbohydrates', 0)} g")

                # Rozwijane okno dla składników
                with st.expander("Składniki", expanded=False):
                    st.markdown(
                        f"<div style='padding: 10px; border-radius: 5px; background-color: transparent;'>"  
                        f"<textarea readonly style='width: 100%; height: 100px; border: none; background: transparent; resize: none;'>{note['text']}</textarea>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                if st.button("Usuń zdjęcie", key=f"remove_{note['id']}"):
                    delete_note_from_db(note['id'])
                    st.rerun()

        # Oddzielenie zdjęć od notatek bez zdjęć
        if notes_without_images:
            st.markdown("<hr>", unsafe_allow_html=True)

        # Wyświetl notatki bez zdjęć
        for note in notes_without_images:
            st.write(f"**Kalorie:** {note['calories']} kcal")
            st.write(f"**Białko:** {note.get('protein', 0)} g")
            st.write(f"**Węglowodany:** {note.get('carbohydrates', 0)} g")

            # Rozwijane okno dla składników
            with st.expander("Składniki", expanded=False):
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: transparent;'>"  
                    f"<textarea readonly style='width: 100%; height: 100px; border: none; background: transparent; resize: none;'>{note['text']}</textarea>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            if st.button("Usuń zdjęcie", key=f"remove_{note['id']}"):
                delete_note_from_db(note['id'])
                st.rerun()
    else:
        st.write("Brak zapisanych zdjęć dla wybranej daty.")