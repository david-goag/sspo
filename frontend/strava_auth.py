import requests
import streamlit as st
from urllib.parse import urlencode


class StravaAuth:
    def __init__(self):
        self.client_id = st.secrets["STRAVA_CLIENT_ID"]
        self.client_secret = st.secrets["STRAVA_CLIENT_SECRET"]
        self.redirect_uri = st.secrets.get(
            "STRAVA_REDIRECT_URI", "https://strava-optimizer.streamlit.app/"
        )
        self.auth_url = "https://www.strava.com/oauth/authorize"
        self.token_url = "https://www.strava.com/oauth/token"
        self.api_url = "https://www.strava.com/api/v3"

    def get_authorization_url(self):
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "approval_prompt": "force",
            "scope": "read,activity:read_all",
        }
        return f"{self.auth_url}?{urlencode(params)}"

    def exchange_code_for_token(self, authorization_code):
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": authorization_code,
            "grant_type": "authorization_code",
        }

        response = requests.post(self.token_url, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting access token: {response.text}")
            return None

    def get_athlete_info(self, access_token):
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.api_url}/athlete", headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting athlete info: {response.text}")
            return None

    def get_activity_stats(self, access_token, athlete_id):
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"{self.api_url}/athletes/{athlete_id}/stats", headers=headers
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting activity stats: {response.text}")
            return None

    def get_activities(self, access_token, per_page=30, page=1):
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"per_page": per_page, "page": page}
        response = requests.get(
            f"{self.api_url}/athlete/activities", headers=headers, params=params
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting activities: {response.text}")
            return None

    def deauthorize(self, access_token):
        data = {"access_token": access_token}
        response = requests.post("https://www.strava.com/oauth/deauthorize", data=data)

        if response.status_code == 200:
            return True
        else:
            st.error(f"Error deauthorizing from Strava: {response.text}")
            return False
