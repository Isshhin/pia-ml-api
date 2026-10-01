import json
import math
from urllib.parse import urlparse

from workers import WorkerEntrypoint, Response


# New model key so this version starts from clean centroids instead of reusing
# centroids that may have moved during earlier manual testing.
MODEL_KEY = "pia_online_kmeans_v2_human_reactions"

# These are normalized "performance strength" dimensions.
FEATURES = [
    "accuracy_strength",
    "attempt_strength",
    "independence_strength",
    "correct_efficiency_strength",
    "correct_streak_strength",
    "wrong_streak_control",
]

# Accuracy and wrong streak are deliberately more influential than speed.
# Speed is only a bonus for correct responses.
FEATURE_WEIGHTS = [
    3.00,  # accuracy
    1.50,  # attempts needed
    1.00,  # independence from hints, only rewarded with accuracy
    0.75,  # speed/efficiency on correct answers only
    1.50,  # correct streak
    2.25,  # control of wrong streak
]

# Cold-start centers for the three clusters.
# These are starting prototypes; they are NOT student labels.
# If online learning is enabled later, the centroids can update incrementally.
DEFAULT_MODEL = {
    "centroids": [
        # Struggling-like performance pattern
        [0.20, 0.35, 0.10, 0.20, 0.10, 0.25],

        # Average-like performance pattern
        [0.60, 0.65, 0.40, 0.55, 0.45, 0.70],

        # Outstanding-like performance pattern
        [0.90, 0.90, 0.85, 0.85, 0.90, 0.95],
    ],

    # Start with a little inertia so accidental test traffic does not move a
    # centroid too aggressively if learning is enabled later.
    "counts": [12, 12, 12],
    "updates": 0,
    "version": 2,
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
        as_float(data, "recent_accuracy", 0.5)
    )

    # 1 attempt is strongest. 5+ attempts reaches the bottom of this feature.
    average_attempts = max(
        1.0,
        as_float(data, "average_attempts", 1.0)
    )
    attempt_strength = 1.0 - clamp(
        (average_attempts - 1.0) / 4.0
    )

    hint_rate = clamp(
        as_float(data, "hint_rate", 0.0)
    )

    # IMPORTANT:
    # "No hint" should not look impressive when the learner is mostly wrong.
    # Independence therefore only becomes strong when it is paired with
    # demonstrated correctness.
    independence_strength = accuracy * (1.0 - hint_rate)

    # Frontend v2 sends an efficiency score calculated ONLY from correct
    # responses. If an older frontend calls this worker, derive a fallback
    # from response time but only when some answers are correct.
    if "correct_response_efficiency" in data:
        correct_efficiency = clamp(
            as_float(data, "correct_response_efficiency", 0.0)
        )
    else:
        average_response_time = max(
            0.0,
            as_float(data, "average_response_time", 120.0)
        )
        correct_efficiency = (
            1.0 / (1.0 + average_response_time / 30.0)
            if accuracy > 0
            else 0.0
        )

    correct_streak_strength = clamp(
        as_float(data, "consecutive_correct", 0.0) / 5.0
    )

    wrong_streak_control = 1.0 - clamp(
        as_float(data, "consecutive_wrong", 0.0) / 5.0
    )

    return [
        accuracy,
        attempt_strength,
        independence_strength,
        correct_efficiency,
        correct_streak_strength,
        wrong_streak_control,
    ]


def weighted_distance(a, b):
    total = 0.0

    for value, center, weight in zip(
        a,
        b,
        FEATURE_WEIGHTS
    ):
        total += weight * ((value - center) ** 2)

    return math.sqrt(total)


def nearest_cluster(vector, centroids):
    distances = [
        weighted_distance(vector, centroid)
        for centroid in centroids
    ]

    cluster = min(
        range(len(distances)),
        key=lambda i: distances[i]
    )

    ordered = sorted(distances)

    if len(ordered) > 1 and ordered[1] > 0:
        confidence = clamp(
            (ordered[1] - ordered[0]) / ordered[1]
        )
    else:
        confidence = 1.0

    return cluster, distances, confidence


def weighted_performance_score(vector):
    numerator = sum(
        value * weight
        for value, weight in zip(vector, FEATURE_WEIGHTS)
    )

    denominator = sum(FEATURE_WEIGHTS)

    return numerator / denominator


def profile_map(centroids):
    # K-Means cluster numbers themselves have no semantic meaning.
    # Order the learned centers from weakest performance pattern to strongest.
    ranked = sorted(
        range(len(centroids)),
        key=lambda i: weighted_performance_score(
            centroids[i]
        )
    )

    return {
        ranked[0]: "struggling",
        ranked[1]: "average",
        ranked[2]: "outstanding",
    }


def update_centroid(model, cluster, vector):
    count = int(
        model["counts"][cluster]
    )

    old = model["centroids"][cluster]
    new_count = count + 1

    model["centroids"][cluster] = [
        old_value
        + ((new_value - old_value) / new_count)

        for old_value, new_value
        in zip(old, vector)
    ]

    model["counts"][cluster] = new_count
    model["updates"] = int(
        model.get("updates", 0)
    ) + 1

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
                "application/json",
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
                    "weighted-online-kmeans-3-profile-v2",
                "profiles": [
                    "struggling",
                    "average",
                    "outstanding",
                ],
                "speed_rule":
                    "speed only rewards correct responses",
                "model_updates":
                    model.get("updates", 0),
            })

        # -----------------------------------------
        # GET /model
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
                    "weighted-online-kmeans-3-profile-v2",
                "updates":
                    model.get("updates", 0),
                "counts":
                    model["counts"],
                "feature_weights":
                    dict(zip(
                        FEATURES,
                        FEATURE_WEIGHTS
                    )),
                "clusters": [
                    {
                        "cluster": i,
                        "profile":
                            mapping[i],
                        "performance_score":
                            round(
                                weighted_performance_score(
                                    model["centroids"][i]
                                ),
                                4
                            ),
                        "centroid":
                            model["centroids"][i],
                    }
                    for i in range(3)
                ],
            })

        # -----------------------------------------
        # POST /predict
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

            should_learn = (
                data.get(
                    "learn",
                    False
                )
                is True
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

                "performance_score":
                    round(
                        weighted_performance_score(
                            vector
                        ),
                        4
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
                ],
            })

        return self.json_response(
            {
                "error":
                    "Not found. Use GET /, GET /model, or POST /predict."
            },
            status=404
        )
