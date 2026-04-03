import spotipy
from spotipy.oauth2 import SpotifyOAuth


CLIENT_ID = "5edb4e879e4b41f8b9a1ac39e52199f3"
CLIENT_SECRET = "f3cb9a20f3d040598db4170042a0effa"
REDIRECT_URI = "http://127.0.0.1:8888/callback"

def get_spotify_client():
    scope = "user-modify-playback-state user-read-playback-state"
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=scope
    ))
    return sp