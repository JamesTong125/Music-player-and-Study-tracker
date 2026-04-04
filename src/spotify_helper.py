import spotipy
from spotipy.oauth2 import SpotifyOAuth

# REPLACE THESE WITH YOUR ACTUAL KEYS
CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"
REDIRECT_URI = "http://localhost:8888/callback"

def get_spotify_client():
    scope = "user-modify-playback-state user-read-playback-state"
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=scope
    ))