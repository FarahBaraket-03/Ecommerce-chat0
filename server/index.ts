// Load environment variables from .env file (must be first import)
import 'dotenv/config'
// Import Express framework for creating web server
import express from "express";
import type { Express, Request, Response } from "express"
// Import MongoDB client for database connection
import { MongoClient } from "mongodb"
// Import our custom AI agent function
import { callAgent } from './agent.js'


const app : Express=express()

import cors from 'cors'

app.use(cors())
app.use(express.json())

const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string)

async function startServer(){
    try{
        await client.connect()
        await client.db('admin').command({ping :1})
        console.log("you successfully connected to mongodb ")
        app.get('/',(req:Request,res:Response)=>{
            res.send('LangGarph Agent Server')
        })

        app.post('/chat',async(req:Request,res:Response)=>{
            const intialMessage =req.body.message
            const threadId =Date.now().toString()
            console.log(intialMessage)
            try{
                const response = await callAgent(client ,intialMessage ,threadId)
                res.json({threadId , response})
            }
            catch(error){
                console.log("error starting conversation",error)
                res.status(500)
            }
        } )


        app.post('/chat/:threadId',async(req:Request,res:Response)=>{
            const {threadId} =req.params
            const {message} =req.body
            if (typeof message !== 'string') {
                res.status(400).json({ error: "Message is required and must be a string." });
                return;
            }
            try{
                const response = await callAgent(client, message, threadId)
                res.json({ response})
            }
            catch(error){
                console.log("error in chat ",error)
                res.status(500)
            }
        })


        const PORT =process.env.PORT || 8000
        app.listen(PORT,()=>{
            console.log(`server runnig on port ${PORT}`)
        })



    }
    catch(error){
        console.log("error in MONGO connecting",error)
        process.exit(1)
    }
}

startServer()