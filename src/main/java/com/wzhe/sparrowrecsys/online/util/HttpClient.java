package com.wzhe.sparrowrecsys.online.util;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.nio.client.CloseableHttpAsyncClient;
import org.apache.http.impl.nio.client.HttpAsyncClients;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Future;

public class HttpClient {
    public static String asyncSinglePostRequest(String host, String body){
        if (null == body || body.isEmpty()){
            return null;
        }

        try {
            final CloseableHttpAsyncClient client = HttpAsyncClients.createDefault();
            client.start();
            HttpEntity bodyEntity = new ByteArrayEntity(body.getBytes(StandardCharsets.UTF_8));
            HttpPost request = new HttpPost(host);
            request.setEntity(bodyEntity);
            final Future<HttpResponse> future = client.execute(request, null);
            final HttpResponse response = future.get();
            return getRespondContent(response);
        }catch (Exception e){
            e.printStackTrace();
            return "";
        }
    }

    public static Map<String, String> asyncMapPostRequest(String host, Map<String, String> bodyMap) throws Exception {
        if (null == bodyMap || bodyMap.isEmpty()){
            return null;
        }

        try {
            final CloseableHttpAsyncClient client = HttpAsyncClients.createDefault();
            client.start();

            HashMap<String, Future<HttpResponse>> futures = new HashMap<>();
            for (Map.Entry<String, String> bodyEntry : bodyMap.entrySet()) {
                String body = bodyEntry.getValue();
                HttpEntity bodyEntity = new ByteArrayEntity(body.getBytes(StandardCharsets.UTF_8));
                HttpPost request = new HttpPost(host);
                request.setEntity(bodyEntity);
                futures.put(bodyEntry.getKey(), client.execute(request, null));
            }

            HashMap<String, String> responds = new HashMap<>();
            for (Map.Entry<String, Future<HttpResponse>> future : futures.entrySet()) {
                final HttpResponse response = future.getValue().get();
                responds.put(future.getKey(), getRespondContent(response));
            }

            client.close();
            return responds;
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }

    public static String getRespondContent(HttpResponse response) throws Exception{
        HttpEntity entity = response.getEntity();
        InputStream is = entity.getContent();
        BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8), 8);
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null)
            sb.append(line).append("\n");
        return sb.toString();
    }

    public static void main(String[] args){
        System.out.println(asyncSinglePostRequest("http://localhost:8501/v1/models/half_plus_two:predict", "{\"instances\": [1.0, 2.0, 5.0]}"));
    }
}
