package com.wzhe.sparrowrecsys.online.service;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wzhe.sparrowrecsys.online.datamanager.DataManager;
import com.wzhe.sparrowrecsys.online.datamanager.User;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 * UserService, return information of a specific user
 */

public class UserService extends HttpServlet {
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws IOException {
        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            //get user id via url parameter
            String userId = request.getParameter("id");

            //get user object from DataManager
            User user = DataManager.getInstance().getUserById(Integer.parseInt(userId));

            //convert movie object to json format and return
            if (null != user) {
                ObjectMapper mapper = new ObjectMapper();
                String jsonUser = mapper.writeValueAsString(user);
                response.getWriter().println(jsonUser);
            }else{
                response.getWriter().println("");
            }

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}
