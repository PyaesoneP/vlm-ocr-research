# realworld_heldout_raw - copy sheet (ground truth)

Write each page below **by hand** on its own sheet, with a dark pen, then photograph or scan it. Save each image using the page id as the filename, for example `rw_101.jpg`.

- Clean pages: **20**  |  Positive pages: **20**  |  Intended errors: **40**
- This is the held-out set. Do not tune prompts, thresholds, guards, or candidate generation after looking at the scores.
- Copy the **Write exactly** text verbatim. For positive pages, that means copying the mistakes exactly as written.
- Use one page per image. Do not resize, crop, or edit the image after capture.

---

## Clean pages (no intended errors)

### rw_101

**Write exactly:**

> The visitor felt embarrassed when he arrived late to the meeting. He apologised quietly and took a seat near the back. The speaker continued without stopping.

### rw_102

**Write exactly:**

> Please keep the blue forms separate from the yellow forms. The boxes are already labelled, so the sorting should be simple. Put the finished stack on my desk.

### rw_103

**Write exactly:**

> The small accident occurred just after lunch. Nobody was hurt, but everyone agreed to be more careful. A short report was written before the office closed.

### rw_104

**Write exactly:**

> There were a lot of chairs in the hall. We moved them into straight rows before the guests arrived. By noon the room looked ready for the ceremony.

### rw_105

**Write exactly:**

> I will definitely call you when the parcel arrives. The shop promised to send a message with the tracking number. Until then, we only have to wait.

### rw_106

**Write exactly:**

> The school will receive new computers next month. Each classroom will get two machines for shared work. The teachers are planning lessons that use them.

### rw_107

**Write exactly:**

> Taking care of the class pet is a serious responsibility. The students must feed it every morning and clean its cage twice a week. Everyone gets a turn.

### rw_108

**Write exactly:**

> At the beginning of the story, the main character is nervous. Later she becomes braver and makes a difficult choice. The ending feels hopeful.

### rw_109

**Write exactly:**

> My friend brought fresh bread to the picnic. We shared it with cheese, fruit, and cold lemonade. The weather stayed warm all afternoon.

### rw_110

**Write exactly:**

> The government announced a new plan for public transport. More buses will run during the morning rush hour. The city hopes this will reduce traffic.

### rw_111

**Write exactly:**

> We are leaving tomorrow before sunrise. Please pack your coat, your notebook, and the map. The train will not wait if we are late.

### rw_112

**Write exactly:**

> It felt weird to walk through the empty school at night. The corridors were silent, and every small sound seemed louder than usual.

### rw_113

**Write exactly:**

> Write your full address clearly on the envelope. If the number is hard to read, the letter may be delayed. Use black ink if possible.

### rw_114

**Write exactly:**

> I stayed home because the rain was heavy. The road outside our house was covered with water. By evening the sky finally began to clear.

### rw_115

**Write exactly:**

> The library closes early on Fridays. Students who need extra time should borrow their books before lunch. The reading room will reopen on Monday.

### rw_116

**Write exactly:**

> Protecting the environment requires many small habits. We can save water, reuse bags, and turn off lights when rooms are empty.

### rw_117

**Write exactly:**

> It is necessary to check your answers before handing in the test. A simple mistake can change the meaning of a whole sentence.

### rw_118

**Write exactly:**

> Regular exercise helps me sleep better. I usually walk in the park after dinner. On rainy days I stretch indoors instead.

### rw_119

**Write exactly:**

> The quiet museum opened at ten o'clock. Visitors walked slowly through the rooms and read the notes beside each painting.

### rw_120

**Write exactly:**

> My cousin plays the piano every Monday. In January she will perform at a small concert near the town centre.

---

## Positive pages (contain intended errors)

### rw_121

**Write exactly (includes the mistakes):**

> The boy felt embarassed after he dropped his lunch tray. He tried to laugh, but his face turned red. Later he sat at a seperate table with his friends.

**Correct version (for reference / annotation):**

> The boy felt embarrassed after he dropped his lunch tray. He tried to laugh, but his face turned red. Later he sat at a separate table with his friends.

**Intended errors:**

- **[SPELL]** "embarassed" -> "embarrassed" (near "...felt embarassed after...")
- **[SPELL]** "seperate" -> "separate" (near "...a seperate table...")

### rw_122

**Write exactly (includes the mistakes):**

> The delay occured during the afternoon storm. A lot of people waited inside the station. The loudspeaker made alot of announcements before the trains moved again.

**Correct version (for reference / annotation):**

> The delay occurred during the afternoon storm. A lot of people waited inside the station. The loudspeaker made a lot of announcements before the trains moved again.

**Intended errors:**

- **[SPELL]** "occured" -> "occurred" (near "...delay occured during...")
- **[MERGE]** "alot" -> "a lot" (near "...made alot of announcements...")

### rw_123

**Write exactly (includes the mistakes):**

> I will definately send the letter today. If you recieve it before Friday, please call me. The envelope has the new address on the front.

**Correct version (for reference / annotation):**

> I will definitely send the letter today. If you receive it before Friday, please call me. The envelope has the new address on the front.

**Intended errors:**

- **[SPELL]** "definately" -> "definitely" (near "...will definately send...")
- **[SPELL]** "recieve" -> "receive" (near "...you recieve it...")

### rw_124

**Write exactly (includes the mistakes):**

> Feeding the dog is my responsability this week. At the begining of each day, I fill his bowl and check his water. He waits by the kitchen door.

**Correct version (for reference / annotation):**

> Feeding the dog is my responsibility this week. At the beginning of each day, I fill his bowl and check his water. He waits by the kitchen door.

**Intended errors:**

- **[SPELL]** "responsability" -> "responsibility" (near "...my responsability this...")
- **[SPELL]** "begining" -> "beginning" (near "...the begining of...")

### rw_125

**Write exactly (includes the mistakes):**

> My best freind moved to another city last year. The goverment built a new road near her neighbourhood. She says the buses are faster now.

**Correct version (for reference / annotation):**

> My best friend moved to another city last year. The government built a new road near her neighbourhood. She says the buses are faster now.

**Intended errors:**

- **[SPELL]** "freind" -> "friend" (near "...best freind moved...")
- **[SPELL]** "goverment" -> "government" (near "...The goverment built...")

### rw_126

**Write exactly (includes the mistakes):**

> We will meet tommorow after school. It feels wierd to plan a picnic in winter, but the forecast says it will be warm. Bring a jacket just in case.

**Correct version (for reference / annotation):**

> We will meet tomorrow after school. It feels weird to plan a picnic in winter, but the forecast says it will be warm. Bring a jacket just in case.

**Intended errors:**

- **[SPELL]** "tommorow" -> "tomorrow" (near "...meet tommorow after...")
- **[SPELL]** "wierd" -> "weird" (near "...feels wierd to...")

### rw_127

**Write exactly (includes the mistakes):**

> Please write your adress at the top of the form. I missed the first bus becuase I could not find my pass. The office opens at nine.

**Correct version (for reference / annotation):**

> Please write your address at the top of the form. I missed the first bus because I could not find my pass. The office opens at nine.

**Intended errors:**

- **[SPELL]** "adress" -> "address" (near "...your adress at...")
- **[SPELL]** "becuase" -> "because" (near "...bus becuase I...")

### rw_128

**Write exactly (includes the mistakes):**

> The libary was closed for repairs all week. A sign near the door said the enviroment inside was unsafe because of dust. Students returned their books online.

**Correct version (for reference / annotation):**

> The library was closed for repairs all week. A sign near the door said the environment inside was unsafe because of dust. Students returned their books online.

**Intended errors:**

- **[SPELL]** "libary" -> "library" (near "...The libary was...")
- **[SPELL]** "enviroment" -> "environment" (near "...the enviroment inside...")

### rw_129

**Write exactly (includes the mistakes):**

> It is neccessary to stretch before running. The coach said that daily excercise would help us avoid injuries. Everyone listened carefully.

**Correct version (for reference / annotation):**

> It is necessary to stretch before running. The coach said that daily exercise would help us avoid injuries. Everyone listened carefully.

**Intended errors:**

- **[SPELL]** "neccessary" -> "necessary" (near "...is neccessary to...")
- **[SPELL]** "excercise" -> "exercise" (near "...daily excercise would...")

### rw_130

**Write exactly (includes the mistakes):**

> I dont know where the spare keys are. We could of checked the drawer before leaving. Now we have to wait outside until Dad gets home.

**Correct version (for reference / annotation):**

> I don't know where the spare keys are. We could have checked the drawer before leaving. Now we have to wait outside until Dad gets home.

**Intended errors:**

- **[SPELL]** "dont" -> "don't" (near "...I dont know...")
- **[GRAMMAR]** "could of" -> "could have" (near "...We could of checked...")

### rw_131

**Write exactly (includes the mistakes):**

> The list of supplies are on the table. Each student needs two pencils and one ruler. The childrens coats should stay on the hooks.

**Correct version (for reference / annotation):**

> The list of supplies is on the table. Each student needs two pencils and one ruler. The children's coats should stay on the hooks.

**Intended errors:**

- **[AGREE]** "are" -> "is" (near "...supplies are on...")
- **[SPELL]** "childrens" -> "children's" (near "...The childrens coats...")

### rw_132

**Write exactly (includes the mistakes):**

> Between you and I, the old plan was easier. There were less mistakes when we used the simple form. The new system still needs work.

**Correct version (for reference / annotation):**

> Between you and me, the old plan was easier. There were fewer mistakes when we used the simple form. The new system still needs work.

**Intended errors:**

- **[GRAMMAR]** "I" -> "me" (near "...you and I,...") — _pronoun case after preposition_
- **[GRAMMAR]** "less" -> "fewer" (near "...were less mistakes...")

### rw_133

**Write exactly (includes the mistakes):**

> The cold weather can effect the battery in your phone. If you loose power, keep the device in your coat pocket. Warmth often helps.

**Correct version (for reference / annotation):**

> The cold weather can affect the battery in your phone. If you lose power, keep the device in your coat pocket. Warmth often helps.

**Intended errors:**

- **[SPELL]** "effect" -> "affect" (near "...can effect the...")
- **[SPELL]** "loose" -> "lose" (near "...you loose power...")

### rw_134

**Write exactly (includes the mistakes):**

> This box is heavier then the last one. It is to large for the shelf, so we should leave it on the floor. Mark it with tape.

**Correct version (for reference / annotation):**

> This box is heavier than the last one. It is too large for the shelf, so we should leave it on the floor. Mark it with tape.

**Intended errors:**

- **[SPELL]** "then" -> "than" (near "...heavier then the...")
- **[SPELL]** "to" -> "too" (near "...It is to large...")

### rw_135

**Write exactly (includes the mistakes):**

> The dog wagged it's tail when the door opened. Your going to like the new park near our street. It has a wide path and several benches.

**Correct version (for reference / annotation):**

> The dog wagged its tail when the door opened. You're going to like the new park near our street. It has a wide path and several benches.

**Intended errors:**

- **[SPELL]** "it's" -> "its" (near "...wagged it's tail...")
- **[HOMOPHONE]** "Your" -> "You're" (near "...Your going to...")

### rw_136

**Write exactly (includes the mistakes):**

> He go to the gym every Saturday morning. If the machines was busy, he runs outside instead. The route around the lake is quiet.

**Correct version (for reference / annotation):**

> He goes to the gym every Saturday morning. If the machines were busy, he runs outside instead. The route around the lake is quiet.

**Intended errors:**

- **[AGREE]** "go" -> "goes" (near "...He go to...")
- **[AGREE]** "was" -> "were" (near "...machines was busy...")

### rw_137

**Write exactly (includes the mistakes):**

> I seen the notice on the classroom door. The teacher done the same activity last year. This time we will work in pairs.

**Correct version (for reference / annotation):**

> I saw the notice on the classroom door. The teacher did the same activity last year. This time we will work in pairs.

**Intended errors:**

- **[GRAMMAR]** "seen" -> "saw" (near "...I seen the...")
- **[GRAMMAR]** "done" -> "did" (near "...teacher done the...")

### rw_138

**Write exactly (includes the mistakes):**

> I no the answer to the first question. She new the answer too, but she waited for someone else to speak. The room stayed silent.

**Correct version (for reference / annotation):**

> I know the answer to the first question. She knew the answer too, but she waited for someone else to speak. The room stayed silent.

**Intended errors:**

- **[HOMOPHONE]** "no" -> "know" (near "...I no the...")
- **[HOMOPHONE]** "new" -> "knew" (near "...She new the...")

### rw_139

**Write exactly (includes the mistakes):**

> We were not aloud to enter the room before the bell. I saved the last peace of cake for my sister. She thanked me after dinner.

**Correct version (for reference / annotation):**

> We were not allowed to enter the room before the bell. I saved the last piece of cake for my sister. She thanked me after dinner.

**Intended errors:**

- **[HOMOPHONE]** "aloud" -> "allowed" (near "...not aloud to...")
- **[HOMOPHONE]** "peace" -> "piece" (near "...last peace of...")

### rw_140

**Write exactly (includes the mistakes):**

> The concert is on monday evening. In january, the same group will visit our school again. We should buy tickets early.

**Correct version (for reference / annotation):**

> The concert is on Monday evening. In January, the same group will visit our school again. We should buy tickets early.

**Intended errors:**

- **[CAPITAL]** "monday" -> "Monday" (near "...on monday evening...")
- **[CAPITAL]** "january" -> "January" (near "...In january, the...")
