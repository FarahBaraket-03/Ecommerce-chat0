import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Ollama } from "@langchain/community/llms/ollama";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages" 
import {
  ChatPromptTemplate,      // For creating structured prompts with placeholders
  MessagesPlaceholder,     // Placeholder for dynamic message history
} from "@langchain/core/prompts"
import { StateGraph } from "@langchain/langgraph"              // State-based workflow orchestration
import { Annotation } from "@langchain/langgraph"              // Type annotations for state management
import { tool } from "@langchain/core/tools"                   // For creating custom tools/functions
import { ToolNode } from "@langchain/langgraph/prebuilt"       // Pre-built node for executing tools
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb" // For saving conversation state
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"   // Vector search integration with MongoDB
import { MongoClient } from "mongodb"                          // MongoDB database client
import { z } from "zod"                                        // Schema validation library
import "dotenv/config" 
import { resolve } from "path";
import type { StringFormatParams } from "zod/v4/core";




//! Utility function to handle API rate limits with exponential backoff
async function retryWithBackoff<T>(fn: () => Promise<T> , maxRetries =3 ):Promise<T>{
    for(let attempt=1 ; attempt<maxRetries ; attempt++){
        try{
            return await fn()
        }
        catch(error){
            if (typeof error === "object" && error !== null && "status" in error && (error as any).status === 429 && attempt < maxRetries){
                const delay =Math.min(1000*Math.pow(2,attempt),30000)
                console.log("rate limit hit .Retrying in seconds")
                await new Promise(resolve => setTimeout(resolve,delay))
                continue
            }
            throw error
        }
    }
    throw new Error("max retries execceded")
}

//^ Main function that creates and runs the AI agent
export async function callAgent(client : MongoClient , query : string , threadId: string){
    try{
        const dbname="inventory_database"
        const db = client.db("inventory_database")
        const collection = await db.createCollection("items")
        const GraphState = Annotation.Root({
            messages : Annotation<BaseMessage[]>(
                {reducer : (x,y) => x.concat(y)}
            )
        })
        const itemLookupTool = tool(
            async({query , n=10})=>{
                try{
                    console.log("item lookup tool called with query : ", query)
                    const totalCount= await collection.countDocuments({})
                    console.log("total document in collection :",totalCount)
                    if(totalCount === 0){
                        console.log("collection is empty")
                        return JSON.stringify({
                            error :"No items found in inventory",
                            message : "the inventory databse appears to be empty",
                            count :0 
                        })
                    }

                    const sampleDocs = await collection.find({}).limit(3).toArray()
                    console.log("sample documents :",sampleDocs)

                    const dbconfig ={
                        collection : collection ,
                        indexName : "vector_index",
                        textKey : "embedding_text",
                        embeddingKey : "embedding"
                    }


                    const vectorStore = new MongoDBAtlasVectorSearch(
                        new OllamaEmbeddings({
                            model: "nomic-embed-text", // good lightweight embedding model
                            baseUrl: "http://localhost:11434" // default Ollama endpoint
                        }),
                            dbconfig
                    )

                    console.log("performing vector seach ....")

                    // Convert query string to embedding vector
                    const queryEmbedding = await vectorStore.embeddings.embedQuery(query);

                    const result = await vectorStore.similaritySearchVectorWithScore(queryEmbedding, n)

                    console.log('vector search returned ', result.length)

                    if(result.length === 0){
                        console.log("vector search returned no results , trying text search .. ")
                        const textResult = await collection.find({
                            $or: [ // OR condition - match any of these fields
                            { item_name: { $regex: query, $options: 'i' } },        
                            { item_description: { $regex: query, $options: 'i' } }, 
                            { categories: { $regex: query, $options: 'i' } },       
                            { embedding_text: { $regex: query, $options: 'i' } }    // Case-insensitive search in embedding text
                            ]
                        }
                        ).limit(n).toArray()

                        console.log("text search returned ",textResult.length)

                        return JSON.stringify({
                            results : textResult,
                            searchType : "text" ,
                            query : query,
                            count : textResult.length
                        })

                    }

                    return JSON.stringify({
                        results :result,
                        searchType : "vector",
                        query : query,
                        count : result.length
                    })


                }
                catch(error){
                    console.error("error in item lookup :" , error)
                    if (typeof error === "object" && error !== null) {
                        console.error("error details : ",{
                            message : (error as any).message,
                            stack : (error as any).stack,
                            name : (error as any).name
                        })
                    } else {
                        console.error("error details : ", error);
                    }
                    return JSON.stringify({
                        error : "failed to search inventory",
                        details : typeof error === "object" && error !== null && "message" in error ? (error as any).message : String(error),
                        query :query
                    })
                }

            },
            {
                name: "itemLookupTool",
                description: "Looks up items in the inventory using vector and text search.",
                schema: z.object({
                    query: z.string().describe("The search query for items."),
                    n: z.number().optional().describe("Number of results to return (default 10).")
                }),
                
            }
        )

        // Array of all available tools (just one in this case)
        const tools = [itemLookupTool];
        // Create a tool execution node for the workflow
        const toolNode = new ToolNode(tools);

        if (!tools || tools.length === 0) {
            throw new Error("Tools array is undefined or empty.");
        } 
            // Ensure tools is not undefined and has at least one element
            const model = new ChatOllama({
                baseUrl: "http://localhost:11434",
                model: "mistral:latest",
                temperature: 0,
                maxRetries: 0,
            })
        


        function shouldContinue(state : typeof GraphState.State){
            const messages= state.messages
            const lastMessage = messages[messages.length - 1]as AIMessage

            if(lastMessage.tool_calls?.length      ){
                return "tools"
            }
            return "__end__"
        }


        async function callModel(state : typeof GraphState.State){
            return retryWithBackoff(async()=>{
                const prompt = ChatPromptTemplate.fromMessages([
          [
            "system", // System message defines the AI's role and behavior
            `You are a helpful E-commerce Chatbot Agent for a furniture store. 
            IMPORTANT: You have access to an item_lookup tool that searches the furniture inventory database. ALWAYS use this tool when customers ask about furniture items, even if the tool returns errors or empty results.
            When using the item_lookup tool:
            - If it returns results, provide helpful details about the furniture items
            - If it returns an error or no results, acknowledge this and offer to help in other ways
            - If the database appears to be empty, let the customer know that inventory might be being updated
            Current time: {time}`,
          ],
          new MessagesPlaceholder("messages"), // Placeholder for conversation history
        ])

        // Fill in the prompt template with actual values
        const formattedPrompt = await prompt.formatMessages({
          time: new Date().toISOString(), // Current timestamp
          messages: state.messages,       // All previous messages
        })

        // Call the AI model with the formatted prompt
        const result = await model.invoke(formattedPrompt)
        // Return new state with the AI's response added
        return { messages: [result] }
            })
        }


        const workflow = new StateGraph (GraphState)
        .addNode("agent" , callModel)
        .addNode('tools',toolNode)
        .addEdge("__start__","agent")
        .addConditionalEdges('agent', shouldContinue)
        .addEdge("tools","agent")

        const checkpointer = new MongoDBSaver({ client, dbName: "inventory_database" })
        const app = workflow.compile({checkpointer})


        const finalState = await app.invoke(
  {
    messages: [
      new HumanMessage(query)
    ]
  },
  {
    recursionLimit: 15,
    configurable: { thread_id: threadId }
  }
);

    const response = finalState.messages[finalState.messages.length-1]?.content

    console.log("agent response :" , response)

    return response

        



    }
    catch(error){
       // Handle different types of errors with user-friendly messages
    if (typeof error === "object" && error !== null && "message" in error) {
      console.error("Error in callAgent:", (error as any).message);
    } else {
      console.error("Error in callAgent:", error);
    }
    
    if (typeof error === "object" && error !== null && "status" in error && (error as any).status === 429) { // Rate limit error
      throw new Error("Service temporarily unavailable due to rate limits. Please try again in a minute.");
    } else if (typeof error === "object" && error !== null && "status" in error && (error as any).status === 401) { // Authentication error
      throw new Error("Authentication failed. Please check your API configuration.");
    } else { // Generic error
      const errorMsg = typeof error === "object" && error !== null && "message" in error ? (error as any).message : String(error);
      throw new Error(`Agent failed: ${errorMsg}`);
    }
    }
}


