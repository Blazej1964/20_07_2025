from io import BytesIO
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
import pyperclip

# Ładowanie danych z pliku .env
env = dotenv_values(".env")

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def get_corrected_text(text):
    instructor_openai_client = get_openai_client()
    res = instructor_openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"Proszę poprawić błędy w tym tekście w dowolnym języku: {text}. Proszę o odpowiedź w tym samym języku, bez komentarzy.",
            },
        ],
    )
    return res.choices[0].message.content.strip()

def get_error_explanation(text):
    instructor_openai_client = get_openai_client()
    res = instructor_openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"Wyszukaj trudne słowa oraz gramatyke (masz do czynienia ze średnio zaawansowanym uczniem), a następnie wyjaśnij ją: {text}",
            },
        ],
    )
    return res.choices[0].message.content.strip()

def generate_audio(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        response_format="mp3",
        input=text
    )
    return response.content

# Ochrona klucza API OpenAI
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI, aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Wprowadź klucz API:", type="password")

        if st.session_state["openai_api_key"]:
            st.success("Klucz jest OK")
#else:
    #st.success("Klucz API załadowany pomyślnie.")

if not st.session_state.get("openai_api_key"):
    st.stop()

client = get_openai_client()

# Sidebar z przyciskami
st.sidebar.title("Nawigacja")
option = st.sidebar.radio("Wybierz opcję:", ["Przetłumacz tekst", "Popraw tekst"])

# Resetowanie stanu session_state przy zmianie opcji w sidebarze
if 'previous_option' not in st.session_state or st.session_state['previous_option'] != option:
    # Czyszczenie pamięci
    for key in list(st.session_state.keys()):
        if key not in ['openai_api_key', 'previous_option']:
            del st.session_state[key]
    st.session_state['previous_option'] = option

# Sekcja do tłumaczenia
if option == "Przetłumacz tekst":
    polski_tekst = st.text_area("**Wprowadź tekst w języku polskim:**", key="polski_tekst_area")
    jezyk = st.selectbox("**Wybierz język do tłumaczenia:**", ["Niemiecki", "Angielski", "Hiszpański"])

    if st.button("Tłumacz"):
        if polski_tekst:
            prompt = f"Tłumacz '{polski_tekst}' na {jezyk} i podaj tylko tłumaczenie:"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            st.session_state.przetlumaczony_tekst = response.choices[0].message.content.strip()
            st.write(f"**Tłumaczenie na język:** {jezyk}")

    # Wyświetlanie przetłumaczonego tekstu
    if "przetlumaczony_tekst" in st.session_state:
        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{st.session_state.przetlumaczony_tekst}</div>", unsafe_allow_html=True)

        # Dodanie przycisku "Kopiuj przetłumaczony tekst"
        if st.button("Kopiuj przetłumaczony tekst"):
            pyperclip.copy(st.session_state.przetlumaczony_tekst)
            st.success("Przetłumaczony tekst został skopiowany do schowka!")

        audio_data = generate_audio(st.session_state.przetlumaczony_tekst)
        audio_bytes = BytesIO(audio_data)
        st.audio(audio_bytes, format='audio/mp3')

        # Dodanie przycisku "Gramatyka"
        if st.button("Wyjaśnienie gramatyki"):
            wyjasnienie_bledow = get_error_explanation(st.session_state.przetlumaczony_tekst)
            st.write("**Wyjaśnienie trudnych słów i gramatyki:**")
            st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{wyjasnienie_bledow}</div>", unsafe_allow_html=True)

            # Dodanie przycisku "Kopiuj wyjaśnienie błędów"
            if st.button("Kopiuj wyjaśnienie błędów"):
                pyperclip.copy(wyjasnienie_bledow)
                st.success("Wyjaśnienie trudnych słów i gramatyki zostało skopiowane do schowka!")

    else:
        st.warning("Proszę wprowadzić tekst do przetłumaczenia.")

# Sekcja do poprawy tekstu
elif option == "Popraw tekst":
    tekst_do_poprawy = st.text_area("Wprowadź tekst do poprawienia:")

    if st.button("Popraw tekst"):
        if tekst_do_poprawy:
            poprawiony_tekst = get_corrected_text(tekst_do_poprawy)
            st.session_state.poprawiony_tekst = poprawiony_tekst
            wyjasnienie_bledow = get_error_explanation(tekst_do_poprawy)
            st.session_state.wyjasnienie_bledow = wyjasnienie_bledow

    if "poprawiony_tekst" in st.session_state:
        st.write("Oto poprawiony tekst:")
        poprawiony_text_area = st.text_area("Poprawiony tekst:", value=st.session_state.poprawiony_tekst, height=80)

        audio_data = generate_audio(st.session_state.poprawiony_tekst)
        audio_bytes = BytesIO(audio_data)
        st.audio(audio_bytes, format='audio/mp3')

        if st.button("Kopiuj poprawiony tekst"):
            pyperclip.copy(poprawiony_text_area)
            st.success("Poprawiony tekst został skopiowany do schowka!")

        st.write("Wyjaśnienie błędów i sugerowane poprawki:")
        wyjasnienie_text_area = st.text_area("Wyjaśnienie błędów:", value=st.session_state.wyjasnienie_bledow, height=200)

        if st.button("Kopiuj wyjaśnienie błędów"):
            pyperclip.copy(wyjasnienie_text_area)
            st.success("Wyjaśnienie błędów zostało skopiowane do schowka!")

    else:
        st.warning("Najpierw popraw tekst, aby go odtworzyć.")