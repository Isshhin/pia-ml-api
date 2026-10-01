import json

from workers import WorkerEntrypoint, Response


class Default(WorkerEntrypoint):

    async def fetch(self, request):

        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }

        # Allow browser requests from thesisv8
        if request.method == "OPTIONS":
            return Response(
                "",
                headers=cors_headers
            )

        # Opening the Worker URL in a browser
        if request.method == "GET":
            return Response(
                json.dumps({
                    "status": "online",
                    "service": "PIA ML API",
                    "message": "Python is running inside Cloudflare."
                }),
                headers={
                    **cors_headers,
                    "Content-Type": "application/json"
                }
            )

        # Receive performance information
        if request.method == "POST":

            data = await request.json()

            return Response(
                json.dumps({
                    "status": "success",

                    # TEMPORARY until we add the actual ML classifier
                    "profile": "average",

                    "received": data
                }),
                headers={
                    **cors_headers,
                    "Content-Type": "application/json"
                }
            )

        return Response(
            "Method not allowed",
            status=405,
            headers=cors_headers
        )
