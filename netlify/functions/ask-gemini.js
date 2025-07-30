// ask-gemini.js の新しいコード

const { GoogleGenerativeAI } = require("@google/generative-ai");
const admin = require('firebase-admin');

// --- 初期設定 ---
// Firebase Admin SDKの初期化（一度だけ実行されるように）
if (admin.apps.length === 0) {
    try {
        // Netlifyの環境変数から設定を読み込む
        const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
        admin.initializeApp({
            credential: admin.credential.cert(serviceAccount)
        });
    } catch (e) {
        console.error("Firebase Admin SDKの初期化に失敗: ", e);
    }
}
const db = admin.firestore();
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });


// --- コサイン類似度を計算するヘルパー関数 ---
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) {
        return 0;
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}


// --- Netlifyのサーバー機能のメイン処理 ---
exports.handler = async (event) => {
    const startTime = Date.now();
    console.log("サーバー機能開始（手動検索モード）");

    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: 'Method Not Allowed' };
    }

    try {
        const { question } = JSON.parse(event.body);
        if (!question) {
            return { statusCode: 400, body: JSON.stringify({ error: "質問がありません。" }) };
        }

        // 1. 質問をベクトル化
        const questionEmbeddingResult = await embeddingModel.embedContent(
            { content: question, taskType: "RETRIEVAL_QUERY" }
        );
        const questionEmbedding = questionEmbeddingResult.embedding.values;

        // 2. Firestoreから【すべて】のプロットベクトルを取得
        const querySnapshot = await db.collection('plot_vectors').get();
        const allPlots = querySnapshot.docs.map(doc => ({
            id: doc.id,
            text: doc.data().text,
            embedding: doc.data().embedding
        }));
        console.log(`[${Date.now() - startTime}ms] ...Firestoreから全 ${allPlots.length} 件のデータを取得`);

        // 3. 各データと質問の類似度を計算
        const similarities = allPlots.map(plot => ({
            text: plot.text,
            score: cosineSimilarity(questionEmbedding, plot.embedding)
        }));

        // 4. 類似度が高い順に並び替え、上位5件を取得
        similarities.sort((a, b) => b.score - a.score);
        const top5Contexts = similarities.slice(0, 5);
        const context = top5Contexts.map(ctx => ctx.text).join("\n\n---\n\n");
        console.log(`[${Date.now() - startTime}ms] ...類似度計算完了。上位5件をコンテキストとして使用。`);

        // 5. 最終的なプロンプトを組み立てて、Proモデルに質問
        const prompt = `あなたはプロの漫画編集者です。提供された以下の「参考情報」にのみ基づいて、ユーザーからの「質問」に回答してください。\n\n# 参考情報\n---\n${context}\n---\n\n# 質問\n${question}`;

        const result = await genAI.getGenerativeModel({ model: "gemini-1.5-flash" }).generateContent(prompt);
        const response = await result.response;
        const text = response.text();

        // 6. AIの回答を返す
        return {
            statusCode: 200,
            body: JSON.stringify({ answer: text })
        };

    } catch (error) {
        console.error("サーバー機能エラー:", error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: "サーバー内部でエラーが発生しました。", details: error.message })
        };
    }
};