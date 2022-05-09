import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Date;

public class test2  {
  private static Connection connect = null;
  private static Statement statement = null;
  private static PreparedStatement preparedStatement = null;
  private static ResultSet resultSet = null;

 public static void main(String[] args) {
    try {
      // Setup the connection with the DB
      connect = DriverManager
          .getConnection("jdbc:mysql://127.0.0.1:3306/test?"
              + "user=YOURUSER&password=YOURPASSWORD"); //TODO replace with your local MySQL user and password

      // Statements allow to issue SQL queries to the database
      statement = connect.createStatement();
      // Result set get the result of the SQL query
      resultSet = statement
          .executeQuery("select * from member");

      writeResultSet(resultSet);

      // PreparedStatements can use variables and are more efficient
      preparedStatement = connect
          .prepareStatement("insert into  member values (?, ?, ?)");
      preparedStatement.setString(1, "Your name"); //TODO type your name
      preparedStatement.setString(2, "Your email"); //TODO type your email
      preparedStatement.setString(3, "Your music band"); //TODO type your favorite music band
      preparedStatement.executeUpdate();
      
      resultSet = statement
          .executeQuery("select * from member");

      writeResultSet(resultSet);
      
      
      System.out.println("Select a table and then print out its content.");
        //TODO call function writeMetaData with the select * from member statement
      
    } catch (Exception e) {
         System.out.println(e);
    } finally {
      close();
    }

  }

  private static void writeMetaData(ResultSet resultSet) throws SQLException {
    //   Now get some metadata from the database
    // Result set get the result of the SQL query
    System.out.println("The columns in the table are: ");
    
    System.out.println("Table: " + resultSet.getMetaData().getTableName(1));
    for  (int i = 1; i<= resultSet.getMetaData().getColumnCount(); i++){
      System.out.println("Column " +i  + " "+ resultSet.getMetaData().getColumnName(i));
    }
  }

  private static void writeResultSet(ResultSet resultSet) throws SQLException {
    // ResultSet is initially before the first data set
    System.out.println("print result from a table..");
    while (resultSet.next()) {
      // It is possible to get the columns via name
      // also possible to get the columns via the column number
      // which starts at 1
      // e.g. resultSet.getSTring(2);
      String name = resultSet.getString("name");
      String email = resultSet.getString("email");
      String band = resultSet.getString("band");
      System.out.println("name: " + name);
      System.out.println("email: " + email);
      System.out.println("band: " + band);
      System.out.println("");
    }
  }

  // You need to close the resultSet
  private static void close() {
    try {
      if (resultSet != null) {
        resultSet.close();
      }

      if (statement != null) {
        statement.close();
      }

      if (connect != null) {
        connect.close();
      }
    } catch (Exception e) {

    }
  }
} 