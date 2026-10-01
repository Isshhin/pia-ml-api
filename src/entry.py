import json
import math
from urllib.parse import urlparse

from workers import WorkerEntrypoint, Response


MODEL_KEY = "pia_online_kmeans_v1"

FEATURES = [
    "recent_accuracy",
    "average_attempts",
    "hint_rate",
    "average_response_time",
    "consecutive_correct",
    "consecutive_wrong",
]


# Cold-start cluster centers.
# These only give the model a sensible starting point.
# As students use the system, the centroids update dynamically.
DEFAULT_MODEL = {
    "centroids": [
        [0.25, 0.30, 0.25, 0.30, 0.15, 0.25],
        [0.58, 0.60, 0.58, 0.60, 0.50, 0.60],
        [0.88, 0.88, 0.90, 0.86, 0.90, 0.90]
    ],
    "counts": [1, 1, 1],
    "updates": 0
}


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def as_float(data, key, default=0.0):
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def feature_vector(data):
    accuracy = clamp(
        as_float(data, "recent_accuracy")
    )

    attempts = clamp(
        (as_float(data, "average_attempts", 1.0) - 1.0) / 4.0
    )

    hints = clamp(
        as_float(data, "hint_rate")
    )

    response_time = clamp(
        as_float(data, "average_response_time") / 120.0
    )

    correct_streak = clamp(
        as_float(data, "consecutive_correct") / 5.0
    )

    wrong_streak = clamp(
        as_float(data, "consecutive_wrong") / 5.0
    )

    # Every dimension is transformed so:
    #
    # higher number = stronger current performance
    #
    # This makes the cluster meanings easier to interpret.
    return [
        accuracy,
        1.0 - attempts,
        1.0 - hints,
        1.0 - response_time,
        correct_streak,
        1.0 - wrong_streak
    ]


def distance(a, b):
    return math.sqrt(
        sum(
            (x - y) ** 2
            for x, y in zip(a, b)
        )
    )


def nearest_cluster(vector, centroids):

    distances = [
        distance(vector, centroid)
        for centroid in centroids
    ]

    cluster = min(
        range(len(distances)),
        key=lambda i: distances[i]
    )

    ordered = sorted(distances)

    if len(ordered) > 1 and ordered[1] > 0:
        confidence = clamp(
            (ordered[1] - ordered[0])
            / ordered[1]
        )
    else:
        confidence = 1.0

    return cluster, distances, confidence


def profile_map(centroids):

    # Cluster numbers themselves have no meaning.
    #
    # Therefore we order the learned clusters
    # from lowest-performing to highest-performing.
    ranked = sorted(
        range(len(centroids)),
        key=lambda i:
            sum(centroids[i])
            / len(centroids[i])
    )

    return {
        ranked[0]: "struggling",
        ranked[1]: "average",
        ranked[2]: "outstanding"
    }


def update_centroid(model, cluster, vector):

    count = int(
        model["counts"][cluster]
    )

    old = model["centroids"][cluster]

    new_count = count + 1

    # Incremental / online K-Means update
    model["centroids"][cluster] = [
        old_value
        + (
            new_value - old_value
        ) / new_count

        for old_value, new_value
        in zip(old, vector)
    ]

    model["counts"][cluster] = new_count

    model["updates"] = (
        int(model.get("updates", 0))
        + 1
    )

    return model


class Default(WorkerEntrypoint):

    def cors_headers(self):

        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods":
                "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers":
                "Content-Type",
            "Content-Type":
                "application/json"
        }


    def json_response(
        self,
        payload,
        status=200
    ):

        return Response(
            json.dumps(payload),
            status=status,
            headers=self.cors_headers()
        )


    async def load_model(self):

        raw = await self.env.ML_MODEL.get(
            MODEL_KEY
        )

        # First ever request:
        # create initial model in Cloudflare KV.
        if raw is None:

            model = json.loads(
                json.dumps(DEFAULT_MODEL)
            )

            await self.env.ML_MODEL.put(
                MODEL_KEY,
                json.dumps(model)
            )

            return model


        try:

            return json.loads(
                str(raw)
            )

        except Exception:

            model = json.loads(
                json.dumps(DEFAULT_MODEL)
            )

            await self.env.ML_MODEL.put(
                MODEL_KEY,
                json.dumps(model)
            )

            return model


    async def save_model(
        self,
        model
    ):

        await self.env.ML_MODEL.put(
            MODEL_KEY,
            json.dumps(model)
        )


    async def fetch(
        self,
        request
    ):

        headers = self.cors_headers()


        # Browser CORS preflight
        if request.method == "OPTIONS":

            return Response(
                "",
                status=204,
                headers=headers
            )


        path = (
            urlparse(request.url)
            .path
            .rstrip("/")
            or "/"
        )


        # -----------------------------------------
        # GET /
        #
        # Check whether the ML service is running.
        # -----------------------------------------
        if (
            request.method == "GET"
            and path == "/"
        ):

            model = await self.load_model()

            return self.json_response({
                "status": "online",
                "service": "PIA ML API",
                "model":
                    "online-kmeans-3-profile",

                "profiles": [
                    "struggling",
                    "average",
                    "outstanding"
                ],

                "model_updates":
                    model.get(
                        "updates",
                        0
                    )
            })


        # -----------------------------------------
        # GET /model
        #
        # Allows us to inspect the learned clusters.
        # -----------------------------------------
        if (
            request.method == "GET"
            and path == "/model"
        ):

            model = await self.load_model()

            mapping = profile_map(
                model["centroids"]
            )

            return self.json_response({

                "model":
                    "online-kmeans-3-profile",

                "updates":
                    model.get(
                        "updates",
                        0
                    ),

                "counts":
                    model["counts"],

                "clusters": [

                    {
                        "cluster": i,

                        "profile":
                            mapping[i],

                        "centroid":
                            model[
                                "centroids"
                            ][i]
                    }

                    for i in range(3)
                ]
            })


        # -----------------------------------------
        # POST /predict
        #
        # Student performance goes here.
        # -----------------------------------------
        if (
            request.method == "POST"
            and path == "/predict"
        ):

            try:

                data = await request.json()

            except Exception:

                return self.json_response(
                    {
                        "error":
                        "Request body must be valid JSON."
                    },
                    status=400
                )


            if not isinstance(data, dict):

                return self.json_response(
                    {
                        "error":
                        "Request body must be a JSON object."
                    },
                    status=400
                )


            vector = feature_vector(
                data
            )

            model = await self.load_model()


            cluster, distances, confidence = (
                nearest_cluster(
                    vector,
                    model["centroids"]
                )
            )


            mapping = profile_map(
                model["centroids"]
            )


            profile = mapping[
                cluster
            ]


            # Default behavior:
            #
            # every meaningful student-performance
            # snapshot helps update the model.
            #
            # Send:
            #
            # "learn": false
            #
            # if you only want prediction.
            should_learn = (
                data.get(
                    "learn",
                    True
                )
                is not False
            )


            if should_learn:

                model = update_centroid(
                    model,
                    cluster,
                    vector
                )

                await self.save_model(
                    model
                )


            return self.json_response({

                "profile":
                    profile,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "cluster":
                    cluster,

                "learned":
                    should_learn,

                "model_updates":
                    model.get(
                        "updates",
                        0
                    ),

                "normalized_features": {

                    FEATURES[i]:
                        round(
                            vector[i],
                            4
                        )

                    for i
                    in range(
                        len(FEATURES)
                    )
                },

                "distances": [
                    round(
                        value,
                        4
                    )
                    for value
                    in distances
                ]
            })


        return self.json_response(
            {
                "error":
                    "Not found. Use GET /, GET /model, or POST /predict."
            },
            status=404
        )
