package com.sparrowrecsys.online.service;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sparrowrecsys.online.datamanager.News;
import com.sparrowrecsys.online.recprocess.SimilarNewsProcess;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

public class SimilarNewsService extends HttpServlet {
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws IOException {
        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            //newsId
            String newsId = request.getParameter("newsId");
            //number of returned movies
            String size = request.getParameter("size");
            //model of calculating similarity, e.g. embedding, graph-embedding
            String model = request.getParameter("model");

            //use SimilarMovieFlow to get similar movies
            List<News> movies = SimilarNewsProcess.getRecList(Integer.parseInt(newsId), Integer.parseInt(size), model);

            //convert movie list to json format and return
            ObjectMapper mapper = new ObjectMapper();
            String jsonMovies = mapper.writeValueAsString(movies);
            response.getWriter().println(jsonMovies);

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }

}
