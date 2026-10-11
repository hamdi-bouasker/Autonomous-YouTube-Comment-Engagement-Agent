import os
from google_auth_oauthlib.flow import InstalledAppFlow

# The required YouTube write scope
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

def main():
    if not os.path.exists("client_secret.json"):
        print("Error: client_secret.json not found in this folder!")
        return

    # Run local server to handle browser OAuth login
    flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0)

    # Save the resulting token into a local 'data' folder
    os.makedirs("data", exist_ok=True)
    token_path = os.path.join("data", "token.json")
    
    with open(token_path, "w") as token:
        token.write(creds.to_json())
    
    print(f"\nSuccess! '{token_path}' has been generated.")
    print("You can now build and run your Docker container.")

if __name__ == "__main__":
    main()