package com.wzhe.sparrowrecsys.online.datamanager;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import java.io.IOException;
import java.util.List;

public class RatingListSerializer extends JsonSerializer<List<Rating>> {

    @Override
    public void serialize(List<Rating> ratingList, JsonGenerator jsonGenerator,
                          SerializerProvider provider) throws IOException {
        jsonGenerator.writeStartArray();
        for (Rating rating : ratingList) {
            jsonGenerator.writeStartObject();
            jsonGenerator.writeObjectField("rating", rating);
            jsonGenerator.writeEndObject();
        }
        jsonGenerator.writeEndArray();
    }
}