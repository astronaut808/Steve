package com.steve.ai.ai;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.steve.ai.SteveMod;
import com.steve.ai.config.SteveConfig;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class OpenAIClient {
    private static final int MAX_RETRIES = 3;
    private static final int INITIAL_RETRY_DELAY_MS = 1000; // 1 second

    private final HttpClient client;
    private final String apiKey;
    private final String apiUrl;

    public OpenAIClient() {
        this.apiKey = SteveConfig.OPENAI_API_KEY.get();
        this.apiUrl = SteveConfig.OPENAI_API_URL.get();
        this.client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    }

    public String sendRequest(String systemPrompt, String userPrompt) {
        JsonObject requestBody = buildRequestBody(systemPrompt, userPrompt);

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl))
            .header("Content-Type", "application/json")
            .timeout(Duration.ofSeconds(60))
            .POST(HttpRequest.BodyPublishers.ofString(requestBody.toString()));

        if (apiKey != null && !apiKey.isEmpty()) {
            requestBuilder.header("Authorization", "Bearer " + apiKey);
        } else {
            SteveMod.LOGGER.info("No OpenAI API key set — assuming local or unauthenticated endpoint.");
        }

        HttpRequest request = requestBuilder.build();

        for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
            try {
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    String responseBody = response.body();
                    if (responseBody == null || responseBody.isEmpty()) {
                        SteveMod.LOGGER.error("OpenAI API returned empty response");
                        return null;
                    }
                    return parseResponse(responseBody);
                }

                // Retryable errors
                if (response.statusCode() == 429 || response.statusCode() >= 500) {
                    if (attempt < MAX_RETRIES - 1) {
                        int delayMs = INITIAL_RETRY_DELAY_MS * (int) Math.pow(2, attempt);
                        SteveMod.LOGGER.warn("API request failed ({}), retrying in {}ms (attempt {}/{})",
                            response.statusCode(), delayMs, attempt + 1, MAX_RETRIES);
                        Thread.sleep(delayMs);
                        continue;
                    }
                }

                SteveMod.LOGGER.error("API request failed: {}", response.statusCode());
                SteveMod.LOGGER.error("Response body: {}", response.body());
                return null;

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                SteveMod.LOGGER.error("Request interrupted", e);
                return null;
            } catch (Exception e) {
                if (attempt < MAX_RETRIES - 1) {
                    int delayMs = INITIAL_RETRY_DELAY_MS * (int) Math.pow(2, attempt);
                    SteveMod.LOGGER.warn("Error communicating with API, retrying in {}ms (attempt {}/{})",
                        delayMs, attempt + 1, MAX_RETRIES, e);
                    try {
                        Thread.sleep(delayMs);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        return null;
                    }
                } else {
                    SteveMod.LOGGER.error("Error communicating with API after {} attempts", MAX_RETRIES, e);
                    return null;
                }
            }
        }

        return null;
    }

    private JsonObject buildRequestBody(String systemPrompt, String userPrompt) {
        JsonObject body = new JsonObject();
        body.addProperty("model", SteveConfig.OPENAI_MODEL.get());
        body.addProperty("temperature", SteveConfig.TEMPERATURE.get());
        body.addProperty("max_tokens", SteveConfig.MAX_TOKENS.get());

        JsonArray messages = new JsonArray();

        JsonObject systemMessage = new JsonObject();
        systemMessage.addProperty("role", "system");
        systemMessage.addProperty("content", systemPrompt);
        messages.add(systemMessage);

        JsonObject userMessage = new JsonObject();
        userMessage.addProperty("role", "user");
        userMessage.addProperty("content", userPrompt);
        messages.add(userMessage);

        body.add("messages", messages);

        return body;
    }

    private String parseResponse(String responseBody) {
        try {
            JsonObject json = JsonParser.parseString(responseBody).getAsJsonObject();

            if (json.has("choices") && json.getAsJsonArray("choices").size() > 0) {
                JsonObject firstChoice = json.getAsJsonArray("choices").get(0).getAsJsonObject();
                if (firstChoice.has("message")) {
                    JsonObject message = firstChoice.getAsJsonObject("message");
                    if (message.has("content")) {
                        return message.get("content").getAsString();
                    }
                }
            }

            SteveMod.LOGGER.error("Unexpected API response format: {}", responseBody);
            return null;

        } catch (Exception e) {
            SteveMod.LOGGER.error("Error parsing API response", e);
            return null;
        }
    }
}
