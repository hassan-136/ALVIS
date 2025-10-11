# ALVIS
A.L.V.I.S. - Adaptive Linguistic Voice Interaction System 
**Track 2: Full-Duplex Interaction**

This track focuses on developing voice dialogue systems that can communicate as naturally as humans — capable of **interrupting, pausing, and responding instantly** without rigid turn-taking. The goal is to evaluate how effectively systems handle **dynamic, real-world conversations** involving interruptions, rejections, and overlapping speech.

The benchmark includes two main scenarios: **Interruption** and **Rejection.**

In **Interruption Scenarios**, the system must handle five types of user behavior:

1. **Follow-up Questions** – User interrupts with a related query; the system must respond immediately.
2. **Negation/Dissatisfaction** – User disagrees mid-response; the system should adapt quickly and correct itself.
3. **Repetition Requests** – User asks to repeat due to unclear audio; the system should restate promptly.
4. **Topic Switching** – User changes the subject; the system must smoothly transition to the new topic.
5. **Silence/Termination** – User requests to stop; the system should pause instantly but remain ready to continue.

In **Rejection Scenarios**, the focus is on distinguishing genuine user input from irrelevant speech:

1. **Real-time Backchannels** – The system ignores short affirmations like “yeah” or “uh-huh.”
2. **Pause Handling** – The system waits patiently during user hesitation.
3. **Third-party Speech** – The system filters out background voices.
4. **Speech Directed at Others** – The system detects when the user speaks to someone else and rejects that input.

Overall, this track aims to **advance speech AI toward truly human-like communication**, emphasizing **responsiveness, contextual understanding, and behavioral rationality** under real conversational conditions.

