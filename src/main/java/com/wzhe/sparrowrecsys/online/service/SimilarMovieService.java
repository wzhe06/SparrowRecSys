package com.wzhe.sparrowrecsys.online.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wzhe.sparrowrecsys.online.datamanager.Movie;
import com.wzhe.sparrowrecsys.online.recflow.SimilarMovieFlow;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

public class SimilarMovieService extends HttpServlet {
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws IOException {
        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            String movieId = request.getParameter("movieId");
            String size = request.getParameter("size");
            String model = request.getParameter("model");

            List<Movie> movies = SimilarMovieFlow.getRecList(Integer.parseInt(movieId), Integer.parseInt(size), model);

            ObjectMapper mapper = new ObjectMapper();
            String jsonMovies = mapper.writeValueAsString(movies);
            response.getWriter().println(jsonMovies);

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}
