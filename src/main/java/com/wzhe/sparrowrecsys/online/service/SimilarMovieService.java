package com.wzhe.sparrowrecsys.online.service;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class SimilarMovieService extends HttpServlet {
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws ServletException,
            IOException {

        StringBuffer result = new StringBuffer();

        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            result.append("{\"simulation\":[");
/*

                for (int i = 0; i < allSimulationId.size(); i++) {

                    result.append("{\"name\":");

                    String simulationName = allSimulationId.get(i);

                    if (simulationNames.containsKey(simulationName)) {
                        simulationName = simulationName + " | " + simulationNames.get(simulationName);
                    }

                    result.append("\"" + simulationName + "\"");
                    result.append(",");
                    result.append("\"value\":");
                    result.append("\"" + allSimulationId.get(i) + "\"");
                    result.append("}");

                    if (i != allSimulationId.size() - 1) {
                        result.append(",");
                    }
                }
*/
            result.append("]}");


        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println(result.toString());
        }

        //System.out.println(result.toString());
        response.getWriter().println(result.toString());

        //System.out.println("access");
    }
}
